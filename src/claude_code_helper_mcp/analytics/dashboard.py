"""DeveloperDashboard -- Comprehensive memory dashboard for developers.

Provides task timeline, decision tree visualization, file modification heat map,
intervention summary, and window state display to give developers complete visibility
into memory system state and patterns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager

logger = logging.getLogger(__name__)


class TaskTimelineEntry:
    """A single entry in the task timeline view."""

    def __init__(
        self,
        ticket_id: str,
        title: str,
        started_at: datetime,
        completed_at: Optional[datetime],
        status: str,
        step_count: int,
    ):
        self.ticket_id = ticket_id
        self.title = title
        self.started_at = started_at
        self.completed_at = completed_at
        self.status = status
        self.step_count = step_count
        self.duration_seconds = 0

        if completed_at is not None:
            self.duration_seconds = (completed_at - started_at).total_seconds()
        else:
            self.duration_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "step_count": self.step_count,
            "duration_seconds": self.duration_seconds,
        }


class DecisionTreeNode:
    """A node in the decision tree visualization."""

    def __init__(self, task_id: str, decision_text: str, reasoning: str, alternatives: list[str]):
        self.task_id = task_id
        self.decision = decision_text
        self.reasoning = reasoning
        self.alternatives = alternatives

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
        }


class FileHeatMapEntry:
    """A file with its modification frequency."""

    def __init__(self, file_path: str, modification_count: int, tasks_modified: list[str]):
        self.file_path = file_path
        self.modification_count = modification_count
        self.tasks_modified = tasks_modified
        self.heat_score = self._calculate_heat_score(modification_count)

    @staticmethod
    def _calculate_heat_score(count: int) -> str:
        """Calculate heat level: cold (1), cool (2-3), warm (4-6), hot (7+)."""
        if count == 1:
            return "cold"
        elif count <= 3:
            return "cool"
        elif count <= 6:
            return "warm"
        else:
            return "hot"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "modification_count": self.modification_count,
            "heat_score": self.heat_score,
            "tasks_modified": self.tasks_modified,
        }


class InterventionSummary:
    """Summary of all detected interventions."""

    def __init__(self):
        self.drift_detections: list[dict] = []
        self.error_loop_detections: list[dict] = []
        self.confusion_detections: list[dict] = []
        self.scope_creep_detections: list[dict] = []
        self.active_interventions: list[dict] = []

    def add_drift(self, ticket_id: str, severity: str, details: str) -> None:
        """Add a drift detection."""
        self.drift_detections.append({
            "ticket_id": ticket_id,
            "severity": severity,
            "details": details,
        })

    def add_error_loop(self, ticket_id: str, action: str, count: int) -> None:
        """Add an error loop detection."""
        self.error_loop_detections.append({
            "ticket_id": ticket_id,
            "action": action,
            "consecutive_failures": count,
        })

    def add_confusion(self, ticket_id: str, confusion_type: str, details: str) -> None:
        """Add a confusion pattern detection."""
        self.confusion_detections.append({
            "ticket_id": ticket_id,
            "confusion_type": confusion_type,
            "details": details,
        })

    def add_scope_creep(self, ticket_id: str, file_path: str, reason: str) -> None:
        """Add a scope creep detection."""
        self.scope_creep_detections.append({
            "ticket_id": ticket_id,
            "file_path": file_path,
            "reason": reason,
        })

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "drift_detections": self.drift_detections,
            "error_loop_detections": self.error_loop_detections,
            "confusion_detections": self.confusion_detections,
            "scope_creep_detections": self.scope_creep_detections,
            "active_interventions": self.active_interventions,
            "total_detections": (
                len(self.drift_detections)
                + len(self.error_loop_detections)
                + len(self.confusion_detections)
                + len(self.scope_creep_detections)
            ),
        }


class WindowStateView:
    """Current state of the sliding window."""

    def __init__(self, window_size: int, completed_tasks: list[str], active_task: Optional[str]):
        self.window_size = window_size
        self.completed_tasks = completed_tasks
        self.active_task = active_task
        self.total_in_window = len(completed_tasks) + (1 if active_task else 0)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "window_size": self.window_size,
            "completed_tasks_in_window": self.completed_tasks,
            "active_task": self.active_task,
            "total_in_window": self.total_in_window,
            "window_occupancy": f"{self.total_in_window}/{self.window_size}",
        }


class DeveloperDashboard:
    """Comprehensive dashboard for developer review of memory system state.

    Provides views of:
    - Task timeline (chronological list with status and metrics)
    - Decision tree (all decisions made with reasoning and alternatives)
    - File modification heat map (most-touched files)
    - Intervention summary (all detected issues and recommendations)
    - Window state (current sliding window status)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize dashboard with a storage path.

        Args:
            storage_path: Path to the .claude-memory directory. If None, auto-detected.
        """
        self.store = MemoryStore(storage_path)
        self.window_manager = WindowManager(storage_path)

        self._timeline: list[TaskTimelineEntry] = []
        self._decision_tree: list[DecisionTreeNode] = []
        self._file_heat_map: list[FileHeatMapEntry] = []
        self._intervention_summary = InterventionSummary()
        self._window_state: Optional[WindowStateView] = None
        self._generated = False

    def generate(self) -> None:
        """Generate all dashboard views by analyzing memory data."""
        logger.info("Generating developer dashboard")

        self._generate_timeline()
        self._generate_decision_tree()
        self._generate_file_heat_map()
        self._generate_intervention_summary()
        self._generate_window_state()

        self._generated = True
        logger.info("Dashboard generation complete")

    def _generate_timeline(self) -> None:
        """Generate chronological task timeline."""
        task_ids = self.store.list_tasks()
        timeline_data: list[tuple[datetime, TaskTimelineEntry]] = []

        for ticket_id in task_ids:
            task = self.store.load_task(ticket_id)
            if task is None:
                logger.warning("Failed to load task %s for timeline", ticket_id)
                continue

            entry = TaskTimelineEntry(
                ticket_id=task.ticket_id,
                title=task.title,
                started_at=task.started_at,
                completed_at=task.completed_at,
                status=task.status.value,
                step_count=len(task.steps),
            )
            timeline_data.append((task.started_at, entry))

        # Sort by start time
        timeline_data.sort(key=lambda x: x[0])
        self._timeline = [entry for _, entry in timeline_data]

    def _generate_decision_tree(self) -> None:
        """Generate decision tree from all task decisions."""
        task_ids = self.store.list_tasks()

        for ticket_id in task_ids:
            task = self.store.load_task(ticket_id)
            if task is None:
                logger.warning("Failed to load task %s for decisions", ticket_id)
                continue

            for decision in task.decisions:
                node = DecisionTreeNode(
                    task_id=task.ticket_id,
                    decision_text=decision.decision,
                    reasoning=decision.reasoning,
                    alternatives=decision.alternatives,
                )
                self._decision_tree.append(node)

    def _generate_file_heat_map(self) -> None:
        """Generate file modification heat map."""
        file_frequency: dict[str, tuple[int, list[str]]] = {}

        task_ids = self.store.list_tasks()
        for ticket_id in task_ids:
            task = self.store.load_task(ticket_id)
            if task is None:
                logger.warning("Failed to load task %s for heat map", ticket_id)
                continue

            for file_record in task.files:
                if file_record.path not in file_frequency:
                    file_frequency[file_record.path] = (0, [])
                count, tasks = file_frequency[file_record.path]
                if ticket_id not in tasks:
                    tasks.append(ticket_id)
                file_frequency[file_record.path] = (count + 1, tasks)

        # Convert to heat map entries and sort by modification count
        heat_map_entries: list[FileHeatMapEntry] = []
        for file_path, (count, tasks) in file_frequency.items():
            entry = FileHeatMapEntry(file_path, count, tasks)
            heat_map_entries.append(entry)

        heat_map_entries.sort(key=lambda e: e.modification_count, reverse=True)
        self._file_heat_map = heat_map_entries

    def _generate_intervention_summary(self) -> None:
        """Generate summary of all detected interventions."""
        task_ids = self.store.list_tasks()

        for ticket_id in task_ids:
            task = self.store.load_task(ticket_id)
            if task is None:
                logger.warning("Failed to load task %s for interventions", ticket_id)
                continue

            # Check for intervention-related metadata
            if "interventions" in task.metadata:
                interventions = task.metadata.get("interventions", {})

                if "drift_report" in interventions:
                    drift = interventions["drift_report"]
                    self._intervention_summary.add_drift(
                        ticket_id,
                        drift.get("severity", "unknown"),
                        drift.get("summary", ""),
                    )

                if "error_loops" in interventions:
                    for error_loop in interventions["error_loops"]:
                        self._intervention_summary.add_error_loop(
                            ticket_id,
                            error_loop.get("action", "unknown"),
                            error_loop.get("failure_count", 0),
                        )

                if "confusions" in interventions:
                    for confusion in interventions["confusions"]:
                        self._intervention_summary.add_confusion(
                            ticket_id,
                            confusion.get("type", "unknown"),
                            confusion.get("details", ""),
                        )

                if "scope_creep" in interventions:
                    for creep in interventions["scope_creep"]:
                        self._intervention_summary.add_scope_creep(
                            ticket_id,
                            creep.get("file_path", "unknown"),
                            creep.get("reason", ""),
                        )

    def _generate_window_state(self) -> None:
        """Generate current sliding window state."""
        window = self.window_manager.window

        completed = []
        if window and window.completed_tasks:
            completed = [task.ticket_id for task in window.completed_tasks]

        active = None
        if window and window.current_task:
            active = window.current_task.ticket_id

        window_size = 3  # Default window size
        if window and window.window_size:
            window_size = window.window_size

        self._window_state = WindowStateView(window_size, completed, active)

    def get_timeline(self) -> list[dict]:
        """Get the task timeline as a list of dictionaries.

        Returns:
            List of timeline entries sorted by start time.
        """
        if not self._generated:
            raise RuntimeError("Call generate() before getting timeline")
        return [entry.to_dict() for entry in self._timeline]

    def get_decision_tree(self) -> list[dict]:
        """Get the decision tree as a list of dictionaries.

        Returns:
            List of decision tree nodes.
        """
        if not self._generated:
            raise RuntimeError("Call generate() before getting decision tree")
        return [node.to_dict() for node in self._decision_tree]

    def get_file_heat_map(self) -> list[dict]:
        """Get the file modification heat map.

        Returns:
            List of file heat map entries sorted by modification count (highest first).
        """
        if not self._generated:
            raise RuntimeError("Call generate() before getting heat map")
        return [entry.to_dict() for entry in self._file_heat_map]

    def get_intervention_summary(self) -> dict:
        """Get the intervention summary.

        Returns:
            Dictionary with all detected interventions organized by type.
        """
        if not self._generated:
            raise RuntimeError("Call generate() before getting intervention summary")
        return self._intervention_summary.to_dict()

    def get_window_state(self) -> dict:
        """Get the current sliding window state.

        Returns:
            Dictionary with window size, tasks, and occupancy.
        """
        if not self._generated:
            raise RuntimeError("Call generate() before getting window state")
        if self._window_state is None:
            return {}
        return self._window_state.to_dict()

    def to_json_dict(self) -> dict:
        """Serialize the complete dashboard to a JSON-compatible dictionary.

        Returns:
            Dictionary with timeline, decision tree, heat map, interventions,
            and window state.
        """
        if not self._generated:
            raise RuntimeError("Call generate() before converting to JSON")

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timeline": self.get_timeline(),
            "decision_tree": self.get_decision_tree(),
            "file_heat_map": self.get_file_heat_map(),
            "interventions": self.get_intervention_summary(),
            "window_state": self.get_window_state(),
            "summary": {
                "total_tasks": len(self._timeline),
                "total_decisions": len(self._decision_tree),
                "files_tracked": len(self._file_heat_map),
                "detections_count": self._intervention_summary.to_dict()["total_detections"],
            },
        }
