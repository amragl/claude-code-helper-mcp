"""Tests for usage burn detection and alerting.

Tests cover:
- UsageRecord / SessionUsage / UsageReport model serialization
- UsageBurnDetector threshold checks (tool calls, steps, time, burst, session)
- Burst detection via sliding window
- Task lifecycle (start, record, complete)
- Persistence to disk
- Integration with InterventionManager

All tests use real computation with in-memory data -- zero mocks.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_code_helper_mcp.models.usage import (
    AlertLevel,
    AlertMetric,
    SessionUsage,
    UsageAlert,
    UsageRecord,
    UsageReport,
)
from claude_code_helper_mcp.detection.usage_burn import (
    UsageBurnDetector,
    DEFAULT_TOOL_CALL_WARN,
    DEFAULT_TOOL_CALL_CRITICAL,
)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestUsageRecord:
    def test_defaults(self):
        rec = UsageRecord(ticket_id="CMH-001")
        assert rec.ticket_id == "CMH-001"
        assert rec.tool_call_count == 0
        assert rec.step_count == 0
        assert rec.files_touched == []
        assert rec.completed_at is None
        assert rec.burst_events == 0
        assert rec.alerts == []

    def test_elapsed_minutes(self):
        rec = UsageRecord(
            ticket_id="CMH-001",
            started_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        assert 9.9 <= rec.elapsed_minutes() <= 10.2

    def test_serialization_roundtrip(self):
        rec = UsageRecord(ticket_id="CMH-001", tool_call_count=42, step_count=10)
        data = rec.model_dump(mode="json")
        restored = UsageRecord.model_validate(data)
        assert restored.ticket_id == "CMH-001"
        assert restored.tool_call_count == 42


class TestSessionUsage:
    def test_add_file_deduplicates(self):
        session = SessionUsage()
        session.add_file("a.py")
        session.add_file("b.py")
        session.add_file("a.py")
        assert session.total_files_touched == ["a.py", "b.py"]

    def test_serialization_roundtrip(self):
        session = SessionUsage(total_tool_calls=100, task_count=3)
        data = session.model_dump(mode="json")
        restored = SessionUsage.model_validate(data)
        assert restored.total_tool_calls == 100
        assert restored.task_count == 3


class TestUsageReport:
    def test_has_alerts_false_when_empty(self):
        report = UsageReport()
        assert not report.has_alerts
        assert report.max_alert_level is None

    def test_has_alerts_true(self):
        report = UsageReport(
            alerts=[
                UsageAlert(
                    level=AlertLevel.WARNING,
                    metric=AlertMetric.TOOL_CALLS,
                    current_value=50,
                    threshold=50,
                    message="test",
                )
            ]
        )
        assert report.has_alerts
        assert report.max_alert_level == AlertLevel.WARNING

    def test_max_alert_level_critical(self):
        report = UsageReport(
            alerts=[
                UsageAlert(
                    level=AlertLevel.WARNING,
                    metric=AlertMetric.TOOL_CALLS,
                    current_value=50,
                    threshold=50,
                    message="warn",
                ),
                UsageAlert(
                    level=AlertLevel.CRITICAL,
                    metric=AlertMetric.BURST,
                    current_value=15,
                    threshold=10,
                    message="crit",
                ),
            ]
        )
        assert report.max_alert_level == AlertLevel.CRITICAL


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------


class TestUsageBurnDetector:
    def test_start_and_complete_task(self):
        detector = UsageBurnDetector()
        rec = detector.start_task("CMH-001")
        assert rec.ticket_id == "CMH-001"
        assert detector.current_record is not None

        completed = detector.complete_task()
        assert completed is not None
        assert completed.completed_at is not None
        assert detector.current_record is None

    def test_record_tool_call_increments(self):
        detector = UsageBurnDetector()
        detector.start_task("CMH-001")
        for _ in range(5):
            detector.record_tool_call()
        assert detector.current_record.tool_call_count == 5
        assert detector.session.total_tool_calls == 5

    def test_record_step_increments(self):
        detector = UsageBurnDetector()
        detector.start_task("CMH-001")
        for _ in range(3):
            detector.record_step()
        assert detector.current_record.step_count == 3

    def test_record_file_deduplicates(self):
        detector = UsageBurnDetector()
        detector.start_task("CMH-001")
        detector.record_file("a.py")
        detector.record_file("b.py")
        detector.record_file("a.py")
        assert detector.current_record.files_touched == ["a.py", "b.py"]

    def test_tool_call_warning_threshold(self):
        detector = UsageBurnDetector(tool_call_warn=5, tool_call_critical=10)
        detector.start_task("CMH-001")
        for _ in range(5):
            detector.record_tool_call()
        report = detector.check_usage()
        assert report.has_alerts
        tool_alerts = [a for a in report.alerts if a.metric == AlertMetric.TOOL_CALLS]
        assert len(tool_alerts) == 1
        assert tool_alerts[0].level == AlertLevel.WARNING

    def test_tool_call_critical_threshold(self):
        detector = UsageBurnDetector(tool_call_warn=5, tool_call_critical=10)
        detector.start_task("CMH-001")
        for _ in range(10):
            detector.record_tool_call()
        report = detector.check_usage()
        tool_alerts = [a for a in report.alerts if a.metric == AlertMetric.TOOL_CALLS]
        assert len(tool_alerts) == 1
        assert tool_alerts[0].level == AlertLevel.CRITICAL

    def test_step_warning_threshold(self):
        detector = UsageBurnDetector(step_warn=3, step_critical=6)
        detector.start_task("CMH-001")
        for _ in range(3):
            detector.record_step()
        report = detector.check_usage()
        step_alerts = [a for a in report.alerts if a.metric == AlertMetric.STEPS]
        assert len(step_alerts) == 1
        assert step_alerts[0].level == AlertLevel.WARNING

    def test_step_critical_threshold(self):
        detector = UsageBurnDetector(step_warn=3, step_critical=6)
        detector.start_task("CMH-001")
        for _ in range(6):
            detector.record_step()
        report = detector.check_usage()
        step_alerts = [a for a in report.alerts if a.metric == AlertMetric.STEPS]
        assert len(step_alerts) == 1
        assert step_alerts[0].level == AlertLevel.CRITICAL

    def test_time_warning_threshold(self):
        detector = UsageBurnDetector(time_warn_minutes=0.0001, time_critical_minutes=999)
        rec = detector.start_task("CMH-001")
        # Force started_at to the past
        rec.started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        detector._current_record = rec
        report = detector.check_usage()
        time_alerts = [a for a in report.alerts if a.metric == AlertMetric.TIME_ELAPSED]
        assert len(time_alerts) == 1
        assert time_alerts[0].level == AlertLevel.WARNING

    def test_time_critical_threshold(self):
        detector = UsageBurnDetector(
            time_warn_minutes=0.0001, time_critical_minutes=0.0002
        )
        rec = detector.start_task("CMH-001")
        rec.started_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        detector._current_record = rec
        report = detector.check_usage()
        time_alerts = [a for a in report.alerts if a.metric == AlertMetric.TIME_ELAPSED]
        assert len(time_alerts) == 1
        assert time_alerts[0].level == AlertLevel.CRITICAL

    def test_burst_detection(self):
        detector = UsageBurnDetector(
            burst_critical=3, burst_window_seconds=60
        )
        detector.start_task("CMH-001")
        # Record 3 calls rapidly
        for _ in range(3):
            detector.record_tool_call()
        report = detector.check_usage()
        burst_alerts = [a for a in report.alerts if a.metric == AlertMetric.BURST]
        assert len(burst_alerts) == 1
        assert burst_alerts[0].level == AlertLevel.CRITICAL
        assert detector.current_record.burst_events == 1

    def test_session_total_calls_critical(self):
        detector = UsageBurnDetector(session_total_calls_critical=5)
        detector.start_task("CMH-001")
        for _ in range(5):
            detector.record_tool_call()
        report = detector.check_usage()
        session_alerts = [
            a for a in report.alerts if a.metric == AlertMetric.SESSION_TOTAL_CALLS
        ]
        assert len(session_alerts) == 1
        assert session_alerts[0].level == AlertLevel.CRITICAL

    def test_no_alerts_below_thresholds(self):
        detector = UsageBurnDetector(
            tool_call_warn=100,
            tool_call_critical=200,
            step_warn=100,
            step_critical=200,
            time_warn_minutes=999,
            time_critical_minutes=9999,
            burst_critical=100,
            session_total_calls_critical=1000,
        )
        detector.start_task("CMH-001")
        for _ in range(5):
            detector.record_tool_call()
            detector.record_step()
        report = detector.check_usage()
        assert not report.has_alerts

    def test_no_record_without_active_task(self):
        detector = UsageBurnDetector()
        # These should be no-ops
        detector.record_tool_call()
        detector.record_step()
        detector.record_file("a.py")
        assert detector.session.total_tool_calls == 0

    def test_auto_complete_on_new_start(self):
        detector = UsageBurnDetector()
        detector.start_task("CMH-001")
        detector.record_tool_call()
        detector.start_task("CMH-002")
        assert detector.current_record.ticket_id == "CMH-002"
        assert detector.session.task_count == 2

    def test_complete_returns_none_without_task(self):
        detector = UsageBurnDetector()
        assert detector.complete_task() is None

    def test_session_accumulates_across_tasks(self):
        detector = UsageBurnDetector()
        detector.start_task("CMH-001")
        for _ in range(3):
            detector.record_tool_call()
        detector.complete_task()

        detector.start_task("CMH-002")
        for _ in range(2):
            detector.record_tool_call()
        detector.complete_task()

        assert detector.session.total_tool_calls == 5
        assert detector.session.task_count == 2


class TestUsageBurnDetectorPersistence:
    def test_task_record_persisted(self, tmp_path):
        storage = tmp_path / ".claude-memory"
        storage.mkdir()
        detector = UsageBurnDetector(storage_path=str(storage))
        detector.start_task("CMH-001")
        detector.record_tool_call()
        detector.complete_task()

        task_file = storage / "usage" / "CMH-001.json"
        assert task_file.exists()
        data = json.loads(task_file.read_text())
        assert data["ticket_id"] == "CMH-001"
        assert data["tool_call_count"] == 1

    def test_session_persisted(self, tmp_path):
        storage = tmp_path / ".claude-memory"
        storage.mkdir()
        detector = UsageBurnDetector(storage_path=str(storage))
        detector.start_task("CMH-001")
        detector.record_tool_call()
        detector.complete_task()

        session_file = storage / "usage" / "session.json"
        assert session_file.exists()
        data = json.loads(session_file.read_text())
        assert data["total_tool_calls"] == 1

    def test_session_loaded_on_init(self, tmp_path):
        storage = tmp_path / ".claude-memory"
        usage_dir = storage / "usage"
        usage_dir.mkdir(parents=True)
        session_data = SessionUsage(total_tool_calls=42, task_count=5)
        (usage_dir / "session.json").write_text(
            json.dumps(session_data.model_dump(mode="json"), default=str)
        )

        detector = UsageBurnDetector(storage_path=str(storage))
        assert detector.session.total_tool_calls == 42
        assert detector.session.task_count == 5


# ---------------------------------------------------------------------------
# Integration: hook simulation
# ---------------------------------------------------------------------------


class TestUsageBurnIntegration:
    def test_simulate_hook_calls_trigger_alerts(self):
        """Simulate the hook calling pattern: check every 10 calls."""
        detector = UsageBurnDetector(
            tool_call_warn=15, tool_call_critical=25
        )
        detector.start_task("CMH-042")

        alerts_seen = []
        for i in range(30):
            detector.record_tool_call()
            detector.record_step()
            # Check every 10 calls (like the hook does)
            if (i + 1) % 10 == 0:
                report = detector.check_usage()
                if report.has_alerts:
                    alerts_seen.extend(report.alerts)

        # At call 10: no alert (below 15)
        # At call 20: warning (>= 15) + steps warning (>= step_warn=30? no, default 30)
        # At call 30: critical tool calls (>= 25)
        tool_alerts = [a for a in alerts_seen if a.metric == AlertMetric.TOOL_CALLS]
        assert any(a.level == AlertLevel.WARNING for a in tool_alerts)
        assert any(a.level == AlertLevel.CRITICAL for a in tool_alerts)

    def test_usage_report_integrates_with_intervention_manager(self):
        """Verify UsageReport flows through InterventionManager."""
        from claude_code_helper_mcp.detection.intervention import InterventionManager

        detector = UsageBurnDetector(tool_call_warn=3, tool_call_critical=5)
        detector.start_task("CMH-001")
        for _ in range(5):
            detector.record_tool_call()
        usage_report = detector.check_usage()
        assert usage_report.has_alerts

        manager = InterventionManager()
        aggregated, response = manager.aggregate_and_respond(
            usage_report=usage_report
        )
        assert aggregated.any_issues_detected
        assert aggregated.usage_report is not None
        assert response.level in ("warning", "escalation")
        assert "usage_burn" in response.detector_details
        assert "Usage burn" in response.evidence_summary
