"""Usage burn detection engine.

Tracks per-task and per-session resource consumption and detects high-burn
scenarios such as excessive tool calls, long-running tasks, and burst activity.
All tracking happens in-memory with optional persistence to disk -- no MCP
tool calls, no agent involvement, no token cost.

Usage::

    detector = UsageBurnDetector(storage_path="/path/to/.claude-memory")
    detector.start_task("CMH-042")
    for _ in range(100):
        detector.record_tool_call()
        detector.record_step()
        report = detector.check_usage()
        if report.has_alerts:
            for alert in report.alerts:
                logger.warning(alert.message)
    detector.complete_task()
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from claude_code_helper_mcp.models.usage import (
    AlertLevel,
    AlertMetric,
    SessionUsage,
    UsageAlert,
    UsageRecord,
    UsageReport,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_TOOL_CALL_WARN = 50
DEFAULT_TOOL_CALL_CRITICAL = 100
DEFAULT_STEP_WARN = 30
DEFAULT_STEP_CRITICAL = 60
DEFAULT_TIME_WARN_MINUTES = 15
DEFAULT_TIME_CRITICAL_MINUTES = 30
DEFAULT_BURST_CRITICAL = 10
DEFAULT_BURST_WINDOW_SECONDS = 60
DEFAULT_SESSION_TOTAL_CALLS_CRITICAL = 500


class UsageBurnDetector:
    """Detects high resource usage ("burn") during agent task execution.

    Parameters
    ----------
    storage_path:
        Path to the ``.claude-memory`` directory.  Usage data is persisted
        under ``<storage_path>/usage/``.  If None, persistence is disabled.
    tool_call_warn:
        Warning threshold for tool calls per task.
    tool_call_critical:
        Critical threshold for tool calls per task.
    step_warn:
        Warning threshold for steps per task.
    step_critical:
        Critical threshold for steps per task.
    time_warn_minutes:
        Warning threshold for task elapsed time (minutes).
    time_critical_minutes:
        Critical threshold for task elapsed time (minutes).
    burst_critical:
        Number of tool calls within burst_window_seconds to trigger burst alert.
    burst_window_seconds:
        Sliding window size for burst detection.
    session_total_calls_critical:
        Critical threshold for cumulative session tool calls.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        tool_call_warn: int = DEFAULT_TOOL_CALL_WARN,
        tool_call_critical: int = DEFAULT_TOOL_CALL_CRITICAL,
        step_warn: int = DEFAULT_STEP_WARN,
        step_critical: int = DEFAULT_STEP_CRITICAL,
        time_warn_minutes: float = DEFAULT_TIME_WARN_MINUTES,
        time_critical_minutes: float = DEFAULT_TIME_CRITICAL_MINUTES,
        burst_critical: int = DEFAULT_BURST_CRITICAL,
        burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
        session_total_calls_critical: int = DEFAULT_SESSION_TOTAL_CALLS_CRITICAL,
    ) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._tool_call_warn = tool_call_warn
        self._tool_call_critical = tool_call_critical
        self._step_warn = step_warn
        self._step_critical = step_critical
        self._time_warn_minutes = time_warn_minutes
        self._time_critical_minutes = time_critical_minutes
        self._burst_critical = burst_critical
        self._burst_window_seconds = burst_window_seconds
        self._session_total_calls_critical = session_total_calls_critical

        # Current task state
        self._current_record: Optional[UsageRecord] = None
        # Sliding window of tool call timestamps for burst detection
        self._call_timestamps: deque[datetime] = deque()

        # Session-level state
        self._session = SessionUsage()

        # Load persisted session if available
        self._load_session()

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def start_task(self, ticket_id: str) -> UsageRecord:
        """Begin tracking a new task.

        If a task is already active, it is completed automatically before
        starting the new one.
        """
        if self._current_record is not None:
            logger.warning(
                "start_task called while task '%s' is active. Auto-completing.",
                self._current_record.ticket_id,
            )
            self.complete_task()

        self._current_record = UsageRecord(ticket_id=ticket_id)
        self._call_timestamps.clear()
        self._session.task_count += 1
        self._session.touch()

        logger.info("Usage tracking started for task %s", ticket_id)
        return self._current_record

    def complete_task(self) -> Optional[UsageRecord]:
        """Complete the current task and persist its record."""
        if self._current_record is None:
            return None

        self._current_record.completed_at = datetime.now(timezone.utc)
        record = self._current_record

        # Persist task record
        self._save_task_record(record)
        self._save_session()

        logger.info(
            "Usage tracking completed for task %s: "
            "%d tool calls, %d steps, %.1f min, %d files, %d burst events",
            record.ticket_id,
            record.tool_call_count,
            record.step_count,
            record.elapsed_minutes(),
            len(record.files_touched),
            record.burst_events,
        )

        self._current_record = None
        self._call_timestamps.clear()
        return record

    # ------------------------------------------------------------------
    # Incremental counters
    # ------------------------------------------------------------------

    def record_tool_call(self) -> None:
        """Record a single tool call for the current task."""
        if self._current_record is None:
            return

        self._current_record.tool_call_count += 1
        self._session.total_tool_calls += 1
        self._session.touch()

        # Track timestamp for burst detection
        self._call_timestamps.append(datetime.now(timezone.utc))

    def record_step(self) -> None:
        """Record a single step for the current task."""
        if self._current_record is None:
            return

        self._current_record.step_count += 1
        self._session.total_steps += 1
        self._session.touch()

    def record_file(self, path: str) -> None:
        """Record a file touched by the current task."""
        if self._current_record is None:
            return

        if path not in self._current_record.files_touched:
            self._current_record.files_touched.append(path)

        self._session.add_file(path)
        self._session.touch()

    # ------------------------------------------------------------------
    # Usage checking
    # ------------------------------------------------------------------

    def check_usage(self) -> UsageReport:
        """Evaluate current usage against thresholds and return a report.

        Returns a UsageReport containing any new alerts.  Alerts are also
        appended to the current task's record.
        """
        alerts: list[UsageAlert] = []

        if self._current_record is not None:
            rec = self._current_record

            # Tool call thresholds
            if rec.tool_call_count >= self._tool_call_critical:
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.CRITICAL,
                        metric=AlertMetric.TOOL_CALLS,
                        current_value=rec.tool_call_count,
                        threshold=self._tool_call_critical,
                        message=(
                            f"CRITICAL: Task {rec.ticket_id} has made "
                            f"{rec.tool_call_count} tool calls "
                            f"(threshold: {self._tool_call_critical})"
                        ),
                    )
                )
            elif rec.tool_call_count >= self._tool_call_warn:
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.WARNING,
                        metric=AlertMetric.TOOL_CALLS,
                        current_value=rec.tool_call_count,
                        threshold=self._tool_call_warn,
                        message=(
                            f"WARNING: Task {rec.ticket_id} has made "
                            f"{rec.tool_call_count} tool calls "
                            f"(threshold: {self._tool_call_warn})"
                        ),
                    )
                )

            # Step thresholds
            if rec.step_count >= self._step_critical:
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.CRITICAL,
                        metric=AlertMetric.STEPS,
                        current_value=rec.step_count,
                        threshold=self._step_critical,
                        message=(
                            f"CRITICAL: Task {rec.ticket_id} has "
                            f"{rec.step_count} steps "
                            f"(threshold: {self._step_critical})"
                        ),
                    )
                )
            elif rec.step_count >= self._step_warn:
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.WARNING,
                        metric=AlertMetric.STEPS,
                        current_value=rec.step_count,
                        threshold=self._step_warn,
                        message=(
                            f"WARNING: Task {rec.ticket_id} has "
                            f"{rec.step_count} steps "
                            f"(threshold: {self._step_warn})"
                        ),
                    )
                )

            # Time thresholds
            elapsed_min = rec.elapsed_minutes()
            if elapsed_min >= self._time_critical_minutes:
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.CRITICAL,
                        metric=AlertMetric.TIME_ELAPSED,
                        current_value=elapsed_min,
                        threshold=self._time_critical_minutes,
                        message=(
                            f"CRITICAL: Task {rec.ticket_id} has been running "
                            f"for {elapsed_min:.1f} min "
                            f"(threshold: {self._time_critical_minutes} min)"
                        ),
                    )
                )
            elif elapsed_min >= self._time_warn_minutes:
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.WARNING,
                        metric=AlertMetric.TIME_ELAPSED,
                        current_value=elapsed_min,
                        threshold=self._time_warn_minutes,
                        message=(
                            f"WARNING: Task {rec.ticket_id} has been running "
                            f"for {elapsed_min:.1f} min "
                            f"(threshold: {self._time_warn_minutes} min)"
                        ),
                    )
                )

            # Burst detection
            burst_count = self._count_burst()
            if burst_count >= self._burst_critical:
                rec.burst_events += 1
                alerts.append(
                    UsageAlert(
                        level=AlertLevel.CRITICAL,
                        metric=AlertMetric.BURST,
                        current_value=burst_count,
                        threshold=self._burst_critical,
                        message=(
                            f"CRITICAL: Burst detected for task {rec.ticket_id}: "
                            f"{burst_count} tool calls in "
                            f"{self._burst_window_seconds}s "
                            f"(threshold: {self._burst_critical})"
                        ),
                    )
                )

            # Append alerts to task record
            rec.alerts.extend(alerts)

        # Session-level checks
        if self._session.total_tool_calls >= self._session_total_calls_critical:
            alerts.append(
                UsageAlert(
                    level=AlertLevel.CRITICAL,
                    metric=AlertMetric.SESSION_TOTAL_CALLS,
                    current_value=self._session.total_tool_calls,
                    threshold=self._session_total_calls_critical,
                    message=(
                        f"CRITICAL: Session total tool calls "
                        f"({self._session.total_tool_calls}) exceeds threshold "
                        f"({self._session_total_calls_critical})"
                    ),
                )
            )

        # Log alerts immediately
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                logger.warning(alert.message)
            else:
                logger.info(alert.message)

        return UsageReport(
            task_record=(
                self._current_record.model_copy() if self._current_record else None
            ),
            session_usage=self._session.model_copy(),
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # Burst detection
    # ------------------------------------------------------------------

    def _count_burst(self) -> int:
        """Count tool calls within the burst window."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self._burst_window_seconds

        # Prune old entries
        while (
            self._call_timestamps
            and self._call_timestamps[0].timestamp() < cutoff
        ):
            self._call_timestamps.popleft()

        return len(self._call_timestamps)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _usage_dir(self) -> Optional[Path]:
        """Return the usage storage directory, creating it if needed."""
        if self._storage_path is None:
            return None
        usage_dir = self._storage_path / "usage"
        usage_dir.mkdir(parents=True, exist_ok=True)
        return usage_dir

    def _save_task_record(self, record: UsageRecord) -> None:
        """Persist a task usage record to disk."""
        usage_dir = self._usage_dir()
        if usage_dir is None:
            return

        path = usage_dir / f"{record.ticket_id}.json"
        try:
            data = record.model_dump(mode="json")
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2, default=str)
                fp.write("\n")
            logger.debug("Saved usage record to %s", path)
        except Exception:
            logger.warning("Failed to save usage record to %s", path, exc_info=True)

    def _save_session(self) -> None:
        """Persist session usage to disk."""
        usage_dir = self._usage_dir()
        if usage_dir is None:
            return

        path = usage_dir / "session.json"
        try:
            data = self._session.model_dump(mode="json")
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2, default=str)
                fp.write("\n")
            logger.debug("Saved session usage to %s", path)
        except Exception:
            logger.warning("Failed to save session usage to %s", path, exc_info=True)

    def _load_session(self) -> None:
        """Load persisted session usage from disk."""
        usage_dir = self._usage_dir()
        if usage_dir is None:
            return

        path = usage_dir / "session.json"
        if not path.is_file():
            return

        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            self._session = SessionUsage.model_validate(data)
            logger.debug("Loaded session usage from %s", path)
        except Exception:
            logger.warning(
                "Failed to load session usage from %s. Starting fresh.",
                path,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_record(self) -> Optional[UsageRecord]:
        """Return the current task's usage record, or None."""
        return self._current_record

    @property
    def session(self) -> SessionUsage:
        """Return the session usage snapshot."""
        return self._session
