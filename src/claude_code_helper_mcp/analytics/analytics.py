"""MemoryAnalytics -- Analyze patterns across all tasks in memory.

Provides insights into task execution patterns by analyzing memory data:
- Average steps and time per ticket
- Frequently modified files across tasks
- Common errors and patterns
- Decision patterns and reasoning trends
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore

logger = logging.getLogger(__name__)


class TaskPattern:
    """Aggregated statistics for a task."""

    def __init__(self, ticket_id: str, title: str):
        self.ticket_id = ticket_id
        self.title = title
        self.step_count = 0
        self.duration: Optional[timedelta] = None
        self.files_modified: list[str] = []
        self.decisions_count = 0
        self.status = "unknown"
        self.errors: list[str] = []

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "step_count": self.step_count,
            "duration_seconds": self.duration.total_seconds() if self.duration else 0,
            "files_modified": self.files_modified,
            "decisions_count": self.decisions_count,
            "status": self.status,
            "errors_count": len(self.errors),
        }


class FileModificationFrequency:
    """Track how often a file is modified across tasks."""

    def __init__(self, path: str):
        self.path = path
        self.modification_count = 0
        self.tasks_that_modified = []

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "path": self.path,
            "modification_count": self.modification_count,
            "tasks": self.tasks_that_modified,
        }


class DecisionPattern:
    """Aggregated decision patterns across tasks."""

    def __init__(self):
        self.total_decisions = 0
        self.decision_types: dict[str, int] = {}
        self.common_alternatives: dict[str, int] = {}
        self.avg_alternatives_per_decision = 0.0

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "total_decisions": self.total_decisions,
            "decision_types": self.decision_types,
            "common_alternatives": self.common_alternatives,
            "avg_alternatives_per_decision": self.avg_alternatives_per_decision,
        }


class MemoryAnalytics:
    """Analyze patterns across all tasks in memory storage.

    Computes aggregate statistics for:
    - Steps and execution time per ticket
    - File modification frequencies
    - Decision patterns and reasoning
    - Error patterns across tasks
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize analytics with a storage path.

        Args:
            storage_path: Path to the .claude-memory directory. If None, auto-detected.
        """
        self.store = MemoryStore(storage_path)
        self._task_patterns: dict[str, TaskPattern] = {}
        self._file_frequencies: dict[str, FileModificationFrequency] = {}
        self._decision_patterns = DecisionPattern()
        self._analyzed = False

    def analyze(self, since: Optional[datetime] = None) -> None:
        """Analyze all tasks in memory, optionally filtering by start time.

        Args:
            since: If provided, only analyze tasks started at or after this time (UTC).
        """
        self._task_patterns.clear()
        self._file_frequencies.clear()
        self._decision_patterns = DecisionPattern()

        # Load all tasks from storage
        task_ids = self.store.list_tasks()
        logger.info("Analyzing %d tasks", len(task_ids))

        for ticket_id in task_ids:
            task = self.store.load_task(ticket_id)
            if task is None:
                logger.warning("Failed to load task %s, skipping", ticket_id)
                continue

            # Filter by since if provided
            if since is not None and task.started_at < since:
                logger.debug("Task %s started before cutoff, skipping", ticket_id)
                continue

            self._analyze_task(task)

        # Post-process decision patterns
        if self._decision_patterns.total_decisions > 0:
            total_alternatives = sum(
                len(alt) for alt in
                [d.alternatives for d in self._collect_all_decisions()]
            )
            self._decision_patterns.avg_alternatives_per_decision = (
                total_alternatives / self._decision_patterns.total_decisions
            )

        self._analyzed = True
        logger.info("Analysis complete: %d tasks analyzed", len(self._task_patterns))

    def _analyze_task(self, task: TaskMemory) -> None:
        """Analyze a single task and update aggregate statistics."""
        pattern = TaskPattern(task.ticket_id, task.title)

        # Step count
        pattern.step_count = len(task.steps)

        # Duration
        if task.completed_at is not None:
            pattern.duration = task.completed_at - task.started_at
        else:
            pattern.duration = datetime.now(timezone.utc) - task.started_at

        # Files modified
        pattern.files_modified = task.get_file_paths()
        for file_path in pattern.files_modified:
            if file_path not in self._file_frequencies:
                self._file_frequencies[file_path] = FileModificationFrequency(file_path)
            freq = self._file_frequencies[file_path]
            freq.modification_count += 1
            if task.ticket_id not in freq.tasks_that_modified:
                freq.tasks_that_modified.append(task.ticket_id)

        # Decisions
        pattern.decisions_count = len(task.decisions)
        self._analyze_decisions(task)

        # Status
        pattern.status = task.status.value

        # Errors (from failed steps)
        for step in task.steps:
            if not step.success and step.description:
                pattern.errors.append(step.description)

        self._task_patterns[task.ticket_id] = pattern

    def _analyze_decisions(self, task: TaskMemory) -> None:
        """Extract decision patterns from a task."""
        for decision in task.decisions:
            self._decision_patterns.total_decisions += 1

            # Extract decision type from the decision text (first few words)
            decision_text = decision.decision.split()[0] if decision.decision else "unknown"
            self._decision_patterns.decision_types[decision_text] = (
                self._decision_patterns.decision_types.get(decision_text, 0) + 1
            )

            # Track alternatives
            for alt in decision.alternatives:
                self._decision_patterns.common_alternatives[alt] = (
                    self._decision_patterns.common_alternatives.get(alt, 0) + 1
                )

    def _collect_all_decisions(self) -> list:
        """Collect all decision records from analyzed tasks."""
        decisions = []
        for ticket_id in self._task_patterns:
            task = self.store.load_task(ticket_id)
            if task:
                decisions.extend(task.decisions)
        return decisions

    def get_summary(self) -> dict:
        """Get a high-level summary of analysis results.

        Returns:
            Dictionary with key statistics.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before getting summary")

        total_tasks = len(self._task_patterns)
        if total_tasks == 0:
            return {
                "total_tasks_analyzed": 0,
                "avg_steps_per_ticket": 0,
                "avg_time_per_ticket_seconds": 0,
                "total_files_modified": 0,
                "top_files": [],
                "total_decisions": 0,
                "status_breakdown": {},
            }

        total_steps = sum(p.step_count for p in self._task_patterns.values())
        total_duration = sum(
            (p.duration or timedelta()).total_seconds()
            for p in self._task_patterns.values()
        )

        status_breakdown = {}
        for pattern in self._task_patterns.values():
            status_breakdown[pattern.status] = status_breakdown.get(pattern.status, 0) + 1

        # Top 10 most modified files
        top_files = sorted(
            self._file_frequencies.values(),
            key=lambda f: f.modification_count,
            reverse=True,
        )[:10]

        return {
            "total_tasks_analyzed": total_tasks,
            "avg_steps_per_ticket": round(total_steps / total_tasks, 2),
            "avg_time_per_ticket_seconds": round(total_duration / total_tasks, 2),
            "total_files_modified": len(self._file_frequencies),
            "top_files": [f.to_dict() for f in top_files],
            "total_decisions": self._decision_patterns.total_decisions,
            "status_breakdown": status_breakdown,
        }

    def get_all_patterns(self) -> dict:
        """Get all task patterns and detailed analysis.

        Returns:
            Dictionary with complete analysis data.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before getting patterns")

        return {
            "tasks": {
                ticket_id: pattern.to_dict()
                for ticket_id, pattern in self._task_patterns.items()
            },
            "files": {
                path: freq.to_dict()
                for path, freq in self._file_frequencies.items()
            },
            "decisions": self._decision_patterns.to_dict(),
        }

    def get_file_modification_heat_map(self) -> dict[str, int]:
        """Get a map of file paths to modification counts.

        Returns:
            Dictionary mapping file path to modification count.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before getting heat map")

        return {
            path: freq.modification_count
            for path, freq in self._file_frequencies.items()
        }

    def get_error_patterns(self) -> dict[str, int]:
        """Aggregate error messages across all tasks.

        Returns:
            Dictionary mapping error description to count.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before getting error patterns")

        error_counts: dict[str, int] = {}
        for pattern in self._task_patterns.values():
            for error in pattern.errors:
                error_counts[error] = error_counts.get(error, 0) + 1

        return dict(
            sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        )

    def get_top_decision_types(self, limit: int = 10) -> dict[str, int]:
        """Get the most common decision types.

        Args:
            limit: Maximum number of decision types to return.

        Returns:
            Dictionary mapping decision type to count.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before getting decision types")

        sorted_types = sorted(
            self._decision_patterns.decision_types.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(sorted_types[:limit])

    def to_json_dict(self) -> dict:
        """Serialize analysis results to a JSON-compatible dictionary.

        Includes summary, all patterns, file heat map, and error patterns.
        """
        if not self._analyzed:
            raise RuntimeError("Call analyze() before converting to JSON")

        return {
            "summary": self.get_summary(),
            "detailed": self.get_all_patterns(),
            "file_heat_map": self.get_file_modification_heat_map(),
            "error_patterns": self.get_error_patterns(),
            "decision_types": self.get_top_decision_types(),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
