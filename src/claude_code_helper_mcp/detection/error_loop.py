"""Error loop detection engine for identifying repeated failure patterns.

The ErrorLoopDetector tracks step outcomes and detects when the same action
(or very similar actions) fails consecutively, indicating the agent may be
stuck in an error loop -- repeating the same failed approach without progress.

This module provides:
- Configurable failure threshold (default 3 consecutive similar failures)
- Action similarity analysis to group related failures
- Discrimination between true error loops and iterative debugging (where
  each attempt makes progress or tries a different approach)
- ErrorLoopReport with severity, evidence, and recommended actions
- Integration with TaskMemory for automatic analysis of recorded steps

Design decisions:
- All analysis is local and deterministic (no external API calls).
- ErrorLoopDetector maintains state across check() calls within a session.
  Call reset() to start a new session.
- Action similarity uses keyword overlap (consistent with AlignmentChecker).
- A "true loop" requires N similar failing actions with no measurable progress
  between them.  "Iterative debugging" is identified when failing actions show
  meaningful variation (different files, different tools, or different approaches).

Depends on:
- CMH-007: StepRecord (action, success fields)
- CMH-009: TaskMemory (steps list, add_step)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from claude_code_helper_mcp.detection.alignment import AlignmentChecker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default number of consecutive similar failures to trigger a loop detection.
DEFAULT_LOOP_THRESHOLD = 3

# Minimum keyword similarity ratio to consider two actions "similar".
SIMILARITY_THRESHOLD = 0.5

# Severity levels for error loop detection.
LOOP_SEVERITY_NONE = "none"
LOOP_SEVERITY_WARNING = "warning"
LOOP_SEVERITY_ACTIVE = "active"
LOOP_SEVERITY_CRITICAL = "critical"

# Recommended actions per severity level.
_RECOMMENDED_ACTIONS = {
    LOOP_SEVERITY_NONE: "No error loop detected. Continue current work.",
    LOOP_SEVERITY_WARNING: (
        "Potential error loop forming. The last 2 actions have failed "
        "similarly. Consider changing your approach or reviewing the error "
        "messages more carefully before retrying."
    ),
    LOOP_SEVERITY_ACTIVE: (
        "Error loop detected. The same action has failed 3+ times "
        "consecutively without meaningful progress. Strongly recommend "
        "pausing, reviewing the root cause, and trying a fundamentally "
        "different approach."
    ),
    LOOP_SEVERITY_CRITICAL: (
        "Critical error loop detected. Repeated failures with no variation "
        "in approach suggest a systemic issue. Recommend: (1) review all "
        "error messages, (2) check preconditions and dependencies, "
        "(3) consider whether the task definition itself needs revision. "
        "An automatic /clear and recovery may be beneficial."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FailureRecord:
    """A single recorded failure event for loop analysis.

    Attributes:
        action: The action description that failed.
        tool_used: The tool that was used (if known).
        error_summary: Summary of the failure result.
        files_involved: Files the action targeted.
        keywords: Extracted keywords from the action (cached for comparisons).
        timestamp: When the failure occurred.
    """

    action: str
    tool_used: str = ""
    error_summary: str = ""
    files_involved: tuple[str, ...] = ()
    keywords: frozenset[str] = field(default_factory=frozenset)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class LoopEvidence:
    """Evidence supporting an error loop detection.

    Attributes:
        consecutive_failures: Number of consecutive similar failures.
        similarity_score: Average pairwise similarity between the failing actions.
        variation_score: How much variation exists between attempts (0.0 = identical
            attempts, 1.0 = completely different approaches each time).
        common_keywords: Keywords shared across all consecutive failures.
        common_files: Files targeted across all consecutive failures.
        common_tools: Tools used across all consecutive failures.
        is_iterative_debugging: True if the failures show meaningful variation
            (suggesting iterative debugging rather than a true loop).
        iteration_progress_signals: List of signals that suggest progress
            between iterations (e.g., different files, different error messages).
    """

    consecutive_failures: int = 0
    similarity_score: float = 0.0
    variation_score: float = 0.0
    common_keywords: list[str] = field(default_factory=list)
    common_files: list[str] = field(default_factory=list)
    common_tools: list[str] = field(default_factory=list)
    is_iterative_debugging: bool = False
    iteration_progress_signals: list[str] = field(default_factory=list)


@dataclass
class ErrorLoopReport:
    """Result of an error loop detection check.

    Attributes:
        severity: One of 'none', 'warning', 'active', 'critical'.
        loop_detected: Boolean shortcut. True when severity is 'active' or 'critical'.
        consecutive_failures: Number of consecutive similar failures observed.
        evidence: Detailed evidence supporting the detection.
        recommended_action: Suggested response based on severity.
        total_failures_in_session: Total failures recorded in the current session.
        total_checks_in_session: Total checks (success + failure) in the session.
        generated_at: UTC timestamp when this report was generated.
    """

    severity: str = LOOP_SEVERITY_NONE
    loop_detected: bool = False
    consecutive_failures: int = 0
    evidence: LoopEvidence = field(default_factory=LoopEvidence)
    recommended_action: str = ""
    total_failures_in_session: int = 0
    total_checks_in_session: int = 0
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "severity": self.severity,
            "loop_detected": self.loop_detected,
            "consecutive_failures": self.consecutive_failures,
            "evidence": {
                "consecutive_failures": self.evidence.consecutive_failures,
                "similarity_score": round(self.evidence.similarity_score, 3),
                "variation_score": round(self.evidence.variation_score, 3),
                "common_keywords": self.evidence.common_keywords,
                "common_files": self.evidence.common_files,
                "common_tools": self.evidence.common_tools,
                "is_iterative_debugging": self.evidence.is_iterative_debugging,
                "iteration_progress_signals": (
                    self.evidence.iteration_progress_signals
                ),
            },
            "recommended_action": self.recommended_action,
            "total_failures_in_session": self.total_failures_in_session,
            "total_checks_in_session": self.total_checks_in_session,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# ErrorLoopDetector
# ---------------------------------------------------------------------------


class ErrorLoopDetector:
    """Detects error loops by tracking consecutive failures of similar actions.

    The detector maintains a session history of step outcomes and analyses
    failure patterns to identify when the agent is stuck in a loop.  Each
    call to :meth:`record_outcome` records a step result, and :meth:`check`
    returns the current loop detection status.

    Parameters
    ----------
    loop_threshold:
        Number of consecutive similar failures to trigger loop detection.
        Default 3.
    similarity_threshold:
        Minimum keyword similarity ratio (0.0-1.0) to consider two actions
        "similar".  Default 0.5.
    variation_threshold:
        Minimum variation score (0.0-1.0) between consecutive failures for
        them to be classified as "iterative debugging" rather than a true
        loop.  Default 0.3.
    """

    def __init__(
        self,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        variation_threshold: float = 0.3,
    ) -> None:
        self._loop_threshold = max(2, loop_threshold)
        self._similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        self._variation_threshold = max(0.0, min(1.0, variation_threshold))

        # Session state.
        self._failure_streak: list[FailureRecord] = []
        self._total_failures: int = 0
        self._total_checks: int = 0
        self._all_outcomes: list[dict] = []  # [{action, success, timestamp}]

    @property
    def total_failures(self) -> int:
        """Return the total number of failures recorded in this session."""
        return self._total_failures

    @property
    def total_checks(self) -> int:
        """Return the total number of checks recorded in this session."""
        return self._total_checks

    @property
    def current_streak(self) -> int:
        """Return the current consecutive failure streak length."""
        return len(self._failure_streak)

    def reset(self) -> None:
        """Clear all session state for a new detection session."""
        self._failure_streak.clear()
        self._total_failures = 0
        self._total_checks = 0
        self._all_outcomes.clear()

    def record_outcome(
        self,
        action: str,
        success: bool,
        tool_used: str = "",
        result_summary: str = "",
        files_involved: Optional[list[str]] = None,
    ) -> ErrorLoopReport:
        """Record a step outcome and return the current loop detection status.

        This is the primary interface.  Call this after each step to record
        whether it succeeded or failed, and receive the current error loop
        assessment.

        Parameters
        ----------
        action:
            Description of the action that was attempted.
        success:
            Whether the action succeeded.
        tool_used:
            Optional name of the tool used.
        result_summary:
            Optional summary of the result (especially useful for failures).
        files_involved:
            Optional list of files the action targeted.

        Returns
        -------
        ErrorLoopReport
            The current error loop detection status.
        """
        self._total_checks += 1
        files = tuple(files_involved) if files_involved else ()
        keywords = frozenset(AlignmentChecker._extract_keywords(action))

        self._all_outcomes.append({
            "action": action,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if success:
            # Success breaks any failure streak.
            self._failure_streak.clear()
            return self._build_report()

        # Record the failure.
        self._total_failures += 1
        failure = FailureRecord(
            action=action,
            tool_used=tool_used,
            error_summary=result_summary,
            files_involved=files,
            keywords=keywords,
        )

        # Check if this failure is similar to the current streak.
        if self._failure_streak:
            if self._is_similar_to_streak(failure):
                self._failure_streak.append(failure)
            else:
                # Different kind of failure -- start a new streak.
                self._failure_streak = [failure]
        else:
            # First failure in a potential streak.
            self._failure_streak.append(failure)

        return self._build_report()

    def check(self) -> ErrorLoopReport:
        """Return the current loop detection status without recording an outcome.

        Useful for querying the current state without adding a new step.

        Returns
        -------
        ErrorLoopReport
            The current error loop detection status.
        """
        return self._build_report()

    def check_task(self, task: "TaskMemory") -> ErrorLoopReport:
        """Analyse a TaskMemory's step history for error loops.

        This is a convenience method that replays the task's steps through
        the detector.  It resets the current session state first.

        Parameters
        ----------
        task:
            A TaskMemory instance whose steps will be analysed.

        Returns
        -------
        ErrorLoopReport
            The error loop assessment based on the task's step history.
        """
        self.reset()
        for step in task.steps:
            self.record_outcome(
                action=step.action,
                success=step.success,
                tool_used=step.tool_used or "",
                result_summary=step.result_summary or "",
                files_involved=step.files_involved,
            )
        return self.check()

    def get_session_summary(self) -> dict:
        """Return a summary of the current detection session.

        Useful for including in reports and monitoring dashboards.
        """
        return {
            "total_checks": self._total_checks,
            "total_failures": self._total_failures,
            "current_failure_streak": len(self._failure_streak),
            "loop_threshold": self._loop_threshold,
            "failure_rate": (
                round(self._total_failures / self._total_checks, 3)
                if self._total_checks > 0
                else 0.0
            ),
            "loop_detected": len(self._failure_streak) >= self._loop_threshold,
        }

    # ------------------------------------------------------------------
    # Internal: similarity analysis
    # ------------------------------------------------------------------

    def _is_similar_to_streak(self, failure: FailureRecord) -> bool:
        """Determine if a failure is similar to the current streak.

        Compares the new failure's keywords against the first failure in the
        streak (the "anchor").  Two failures are considered similar if their
        keyword overlap ratio meets the similarity threshold.
        """
        if not self._failure_streak:
            return True

        anchor = self._failure_streak[0]
        return self._compute_similarity(anchor.keywords, failure.keywords) >= self._similarity_threshold

    @staticmethod
    def _compute_similarity(
        keywords_a: frozenset[str],
        keywords_b: frozenset[str],
    ) -> float:
        """Compute Jaccard-like similarity between two keyword sets.

        Returns the ratio of intersection to union.  If both sets are empty,
        returns 1.0 (trivially similar).
        """
        if not keywords_a and not keywords_b:
            return 1.0
        if not keywords_a or not keywords_b:
            return 0.0

        intersection = keywords_a & keywords_b
        union = keywords_a | keywords_b
        return len(intersection) / len(union)

    # ------------------------------------------------------------------
    # Internal: variation and iterative debugging detection
    # ------------------------------------------------------------------

    def _compute_evidence(self) -> LoopEvidence:
        """Analyse the current failure streak and produce evidence.

        Computes similarity, variation, common elements, and determines
        whether the pattern is a true loop or iterative debugging.
        """
        streak = self._failure_streak

        if len(streak) < 2:
            return LoopEvidence(
                consecutive_failures=len(streak),
            )

        # Compute average pairwise similarity.
        similarity_sum = 0.0
        pair_count = 0
        for i in range(len(streak)):
            for j in range(i + 1, len(streak)):
                similarity_sum += self._compute_similarity(
                    streak[i].keywords, streak[j].keywords
                )
                pair_count += 1

        avg_similarity = similarity_sum / pair_count if pair_count > 0 else 0.0

        # Compute variation score.
        variation_score = self._compute_variation_score(streak)

        # Find common keywords across all failures.
        if streak:
            common_kw = set(streak[0].keywords)
            for rec in streak[1:]:
                common_kw &= set(rec.keywords)
            common_keywords = sorted(common_kw)[:10]
        else:
            common_keywords = []

        # Find common files.
        all_file_sets = [set(rec.files_involved) for rec in streak if rec.files_involved]
        if all_file_sets:
            common_files_set = all_file_sets[0]
            for fs in all_file_sets[1:]:
                common_files_set &= fs
            common_files = sorted(common_files_set)[:10]
        else:
            common_files = []

        # Find common tools.
        tools = [rec.tool_used for rec in streak if rec.tool_used]
        tool_counter = Counter(tools)
        common_tools = [
            tool for tool, count in tool_counter.most_common(5)
            if count >= len(streak) * 0.5
        ]

        # Detect iterative debugging.
        progress_signals = self._detect_progress_signals(streak)
        is_iterative = (
            variation_score >= self._variation_threshold
            and len(progress_signals) > 0
        )

        return LoopEvidence(
            consecutive_failures=len(streak),
            similarity_score=avg_similarity,
            variation_score=variation_score,
            common_keywords=common_keywords,
            common_files=common_files,
            common_tools=common_tools,
            is_iterative_debugging=is_iterative,
            iteration_progress_signals=progress_signals,
        )

    @staticmethod
    def _compute_variation_score(streak: list[FailureRecord]) -> float:
        """Compute how much variation exists between consecutive failures.

        A score of 0.0 means all failures are essentially identical.
        A score of 1.0 means every failure targets different files, uses
        different tools, and has different error summaries.

        We measure variation across three dimensions:
        - File diversity: how many unique file sets appear
        - Tool diversity: how many unique tools are used
        - Error diversity: how many unique error summaries appear
        """
        if len(streak) < 2:
            return 0.0

        n = len(streak)

        # File diversity: ratio of unique file sets to total entries.
        file_sets = [rec.files_involved for rec in streak]
        unique_file_sets = len(set(file_sets))
        file_diversity = (unique_file_sets - 1) / max(1, n - 1)

        # Tool diversity.
        tools = [rec.tool_used for rec in streak]
        unique_tools = len(set(tools))
        tool_diversity = (unique_tools - 1) / max(1, n - 1)

        # Error summary diversity (using keyword extraction for robustness).
        error_kw_sets = []
        for rec in streak:
            kw = frozenset(AlignmentChecker._extract_keywords(rec.error_summary))
            error_kw_sets.append(kw)
        unique_error_patterns = len(set(error_kw_sets))
        error_diversity = (unique_error_patterns - 1) / max(1, n - 1)

        # Weighted combination: files 40%, errors 40%, tools 20%.
        variation = (
            0.40 * file_diversity
            + 0.40 * error_diversity
            + 0.20 * tool_diversity
        )
        return max(0.0, min(1.0, variation))

    @staticmethod
    def _detect_progress_signals(streak: list[FailureRecord]) -> list[str]:
        """Detect signals that suggest progress between failure iterations.

        Returns a list of human-readable descriptions of progress signals.
        An empty list means no progress is detected (true loop).
        """
        if len(streak) < 2:
            return []

        signals: list[str] = []

        # Signal 1: Files are changing between attempts.
        file_changes = 0
        for i in range(1, len(streak)):
            if streak[i].files_involved != streak[i - 1].files_involved:
                file_changes += 1
        if file_changes > 0:
            signals.append(
                f"Files changed between {file_changes} of "
                f"{len(streak) - 1} consecutive attempts."
            )

        # Signal 2: Tools are changing between attempts.
        tool_changes = 0
        for i in range(1, len(streak)):
            if streak[i].tool_used != streak[i - 1].tool_used:
                tool_changes += 1
        if tool_changes > 0:
            signals.append(
                f"Tools changed between {tool_changes} of "
                f"{len(streak) - 1} consecutive attempts."
            )

        # Signal 3: Error messages are changing (suggesting the error is evolving).
        error_changes = 0
        for i in range(1, len(streak)):
            prev_kw = AlignmentChecker._extract_keywords(streak[i - 1].error_summary)
            curr_kw = AlignmentChecker._extract_keywords(streak[i].error_summary)
            if prev_kw and curr_kw:
                union = prev_kw | curr_kw
                if union:
                    overlap_ratio = len(prev_kw & curr_kw) / len(union)
                    if overlap_ratio < 0.7:
                        error_changes += 1
        if error_changes > 0:
            signals.append(
                f"Error messages changed meaningfully between {error_changes} "
                f"of {len(streak) - 1} consecutive attempts."
            )

        # Signal 4: Action descriptions show keyword variation.
        action_changes = 0
        for i in range(1, len(streak)):
            if streak[i].keywords and streak[i - 1].keywords:
                union = streak[i].keywords | streak[i - 1].keywords
                if union:
                    new_keywords = streak[i].keywords - streak[i - 1].keywords
                    if len(new_keywords) >= 2:
                        action_changes += 1
        if action_changes > 0:
            signals.append(
                f"Action descriptions introduced new keywords in "
                f"{action_changes} of {len(streak) - 1} consecutive attempts."
            )

        return signals

    # ------------------------------------------------------------------
    # Internal: report building
    # ------------------------------------------------------------------

    def _build_report(self) -> ErrorLoopReport:
        """Build the current ErrorLoopReport from session state."""
        streak_len = len(self._failure_streak)
        evidence = self._compute_evidence()
        severity = self._classify_severity(streak_len, evidence)
        recommended = _RECOMMENDED_ACTIONS.get(severity, "")
        loop_detected = severity in (LOOP_SEVERITY_ACTIVE, LOOP_SEVERITY_CRITICAL)

        return ErrorLoopReport(
            severity=severity,
            loop_detected=loop_detected,
            consecutive_failures=streak_len,
            evidence=evidence,
            recommended_action=recommended,
            total_failures_in_session=self._total_failures,
            total_checks_in_session=self._total_checks,
        )

    def _classify_severity(
        self,
        streak_len: int,
        evidence: LoopEvidence,
    ) -> str:
        """Classify the error loop severity.

        Severity depends on:
        1. The length of the consecutive failure streak
        2. Whether the pattern is iterative debugging (lower severity)
        3. Whether the streak meets or exceeds the configured threshold
        """
        if streak_len == 0:
            return LOOP_SEVERITY_NONE

        if streak_len == 1:
            return LOOP_SEVERITY_NONE

        # If iterative debugging is detected, reduce severity by one level.
        if evidence.is_iterative_debugging:
            if streak_len < self._loop_threshold:
                return LOOP_SEVERITY_NONE
            elif streak_len < self._loop_threshold + 2:
                return LOOP_SEVERITY_WARNING
            else:
                return LOOP_SEVERITY_ACTIVE

        # Standard severity classification.
        if streak_len < self._loop_threshold - 1:
            return LOOP_SEVERITY_NONE
        elif streak_len == self._loop_threshold - 1:
            return LOOP_SEVERITY_WARNING
        elif streak_len < self._loop_threshold + 3:
            return LOOP_SEVERITY_ACTIVE
        else:
            return LOOP_SEVERITY_CRITICAL
