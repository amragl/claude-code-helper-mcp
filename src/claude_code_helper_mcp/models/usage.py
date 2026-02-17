"""Pydantic models for usage burn tracking and alerting.

Provides data models for per-task usage records, cumulative session usage,
usage alerts, and usage reports.  These models are used by the
:class:`~claude_code_helper_mcp.detection.usage_burn.UsageBurnDetector` to
track resource consumption and detect high-burn scenarios.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AlertLevel(str, Enum):
    """Severity level for a usage alert."""

    WARNING = "warning"
    CRITICAL = "critical"


class AlertMetric(str, Enum):
    """The metric that triggered an alert."""

    TOOL_CALLS = "tool_calls"
    STEPS = "steps"
    TIME_ELAPSED = "time_elapsed"
    BURST = "burst"
    SESSION_TOTAL_CALLS = "session_total_calls"


class UsageAlert(BaseModel):
    """A single usage threshold alert.

    Attributes
    ----------
    level:
        Alert severity (warning or critical).
    metric:
        The metric that triggered the alert.
    current_value:
        The current value of the metric.
    threshold:
        The threshold that was exceeded.
    message:
        Human-readable alert message.
    timestamp:
        When the alert was generated.
    """

    level: AlertLevel
    metric: AlertMetric
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class UsageRecord(BaseModel):
    """Per-task usage metrics.

    Tracks resource consumption for a single task lifecycle
    (from start_task to complete_task).

    Attributes
    ----------
    ticket_id:
        The ticket identifier for this task.
    tool_call_count:
        Total tool calls made during this task.
    step_count:
        Total steps recorded during this task.
    files_touched:
        Set of file paths touched during this task.
    started_at:
        When the task was started.
    completed_at:
        When the task was completed (None if still active).
    burst_events:
        Number of burst events detected during this task.
    alerts:
        List of alerts generated during this task.
    """

    ticket_id: str
    tool_call_count: int = 0
    step_count: int = 0
    files_touched: list[str] = Field(default_factory=list)
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    burst_events: int = 0
    alerts: list[UsageAlert] = Field(default_factory=list)

    def elapsed_seconds(self) -> float:
        """Return elapsed time in seconds since task start."""
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()

    def elapsed_minutes(self) -> float:
        """Return elapsed time in minutes since task start."""
        return self.elapsed_seconds() / 60.0


class SessionUsage(BaseModel):
    """Cumulative session-level usage metrics.

    Tracks aggregate resource consumption across all tasks in a session.

    Attributes
    ----------
    total_tool_calls:
        Total tool calls across all tasks in this session.
    total_steps:
        Total steps across all tasks.
    total_files_touched:
        Unique files touched across all tasks.
    task_count:
        Number of tasks started in this session.
    started_at:
        When the session began.
    last_activity:
        Timestamp of the most recent activity.
    """

    total_tool_calls: int = 0
    total_steps: int = 0
    total_files_touched: list[str] = Field(default_factory=list)
    task_count: int = 0
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def add_file(self, path: str) -> None:
        """Add a file path to the session's touched files (deduplicates)."""
        if path not in self.total_files_touched:
            self.total_files_touched.append(path)

    def touch(self) -> None:
        """Update last_activity to now."""
        self.last_activity = datetime.now(timezone.utc)


class UsageReport(BaseModel):
    """Report from a usage check containing any triggered alerts.

    Attributes
    ----------
    task_record:
        Current task usage record snapshot.
    session_usage:
        Current session usage snapshot.
    alerts:
        List of new alerts from this check.
    generated_at:
        When this report was generated.
    """

    task_record: Optional[UsageRecord] = None
    session_usage: Optional[SessionUsage] = None
    alerts: list[UsageAlert] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def has_alerts(self) -> bool:
        """Return True if any alerts were triggered."""
        return len(self.alerts) > 0

    @property
    def max_alert_level(self) -> Optional[AlertLevel]:
        """Return the highest alert level, or None if no alerts."""
        if not self.alerts:
            return None
        if any(a.level == AlertLevel.CRITICAL for a in self.alerts):
            return AlertLevel.CRITICAL
        return AlertLevel.WARNING
