"""CrossProjectMemory -- Aggregate memory data across multiple Agent Forge projects.

Enables cross-project analytics by discovering projects via the Agent Forge hub
registry and aggregating memory state from each project's .claude-memory/ storage.

Provides:
- Project discovery from hub registry
- Aggregated analytics across projects
- Global memory storage in ~/.claude-memory-global/
- Cross-project insights and patterns
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from claude_code_helper_mcp.analytics.analytics import MemoryAnalytics
from claude_code_helper_mcp.storage.store import MemoryStore

logger = logging.getLogger(__name__)

# Global storage directory for cross-project memory state
GLOBAL_STORAGE_DIR = ".claude-memory-global"


class ProjectMemorySnapshot:
    """Memory snapshot for a single project."""

    def __init__(self, project_id: str, project_name: str, local_path: str):
        self.project_id = project_id
        self.project_name = project_name
        self.local_path = local_path
        self.total_tasks = 0
        self.total_steps = 0
        self.total_decisions = 0
        self.top_files: list[dict] = []
        self.error_patterns: dict[str, int] = {}
        self.avg_steps_per_task = 0.0
        self.last_scanned: Optional[datetime] = None
        self.accessible = False
        self.scan_error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "local_path": self.local_path,
            "total_tasks": self.total_tasks,
            "total_steps": self.total_steps,
            "total_decisions": self.total_decisions,
            "top_files": self.top_files,
            "error_patterns": self.error_patterns,
            "avg_steps_per_task": self.avg_steps_per_task,
            "last_scanned": self.last_scanned.isoformat() if self.last_scanned else None,
            "accessible": self.accessible,
            "scan_error": self.scan_error,
        }


class CrossProjectAnalytics:
    """Aggregated analytics across multiple projects."""

    def __init__(self):
        self.total_projects_scanned = 0
        self.total_projects_accessible = 0
        self.total_tasks_across_projects = 0
        self.total_steps_across_projects = 0
        self.total_decisions_across_projects = 0
        self.global_top_files: dict[str, int] = {}
        self.global_error_patterns: dict[str, int] = {}
        self.project_snapshots: dict[str, ProjectMemorySnapshot] = {}
        self.scanned_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "total_projects_scanned": self.total_projects_scanned,
            "total_projects_accessible": self.total_projects_accessible,
            "total_tasks_across_projects": self.total_tasks_across_projects,
            "total_steps_across_projects": self.total_steps_across_projects,
            "total_decisions_across_projects": self.total_decisions_across_projects,
            "global_top_files": self.global_top_files,
            "global_error_patterns": self.global_error_patterns,
            "project_snapshots": {
                pid: snapshot.to_dict()
                for pid, snapshot in self.project_snapshots.items()
            },
            "scanned_at": self.scanned_at.isoformat() if self.scanned_at else None,
        }


class CrossProjectMemory:
    """Aggregate memory data across multiple Agent Forge projects.

    Discovers projects from the Agent Forge hub registry, scans each project's
    .claude-memory/ directory, and aggregates analytics across all projects.
    Maintains global state in ~/.claude-memory-global/.

    Typical usage::

        cross_project = CrossProjectMemory()
        analytics = cross_project.scan_all_projects()
        print(analytics.total_projects_accessible)
        cross_project.save_analytics(analytics)
    """

    def __init__(self, hub_registry_path: Optional[str] = None,
                 global_storage_path: Optional[str] = None):
        """Initialize CrossProjectMemory.

        Args:
            hub_registry_path: Path to the Agent Forge hub registry JSON.
                If None, searches common locations.
            global_storage_path: Path to global storage directory.
                If None, uses ~/.claude-memory-global/.
        """
        self.hub_registry_path = self._resolve_hub_registry(hub_registry_path)
        self.global_storage_path = self._resolve_global_storage(global_storage_path)
        self._ensure_global_storage()

    def scan_all_projects(self) -> CrossProjectAnalytics:
        """Scan all registered projects and aggregate their memory data.

        Reads the hub registry, discovers all projects, and for each project:
        1. Checks if local_path is accessible
        2. Loads .claude-memory/ data if it exists
        3. Computes analytics for that project
        4. Aggregates into global statistics

        Returns:
            CrossProjectAnalytics with complete cross-project data.
        """
        analytics = CrossProjectAnalytics()
        projects = self._load_projects_from_registry()

        for project in projects:
            analytics.total_projects_scanned += 1
            snapshot = self._scan_project(project)
            analytics.project_snapshots[project["id"]] = snapshot

            if snapshot.accessible:
                analytics.total_projects_accessible += 1
                analytics.total_tasks_across_projects += snapshot.total_tasks
                analytics.total_steps_across_projects += snapshot.total_steps
                analytics.total_decisions_across_projects += (
                    snapshot.total_decisions
                )

                # Merge top files
                for file_info in snapshot.top_files:
                    file_path = file_info["path"]
                    count = file_info["modification_count"]
                    analytics.global_top_files[file_path] = (
                        analytics.global_top_files.get(file_path, 0) + count
                    )

                # Merge error patterns
                for error, count in snapshot.error_patterns.items():
                    analytics.global_error_patterns[error] = (
                        analytics.global_error_patterns.get(error, 0) + count
                    )

        # Sort global top files and error patterns
        analytics.global_top_files = dict(
            sorted(
                analytics.global_top_files.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
        )
        analytics.global_error_patterns = dict(
            sorted(
                analytics.global_error_patterns.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
        )

        analytics.scanned_at = datetime.now(timezone.utc)
        return analytics

    def scan_projects(self, project_ids: list[str]) -> CrossProjectAnalytics:
        """Scan specific projects by their IDs.

        Args:
            project_ids: List of project IDs to scan (e.g., ["CMH", "snow-csa-agent"]).

        Returns:
            CrossProjectAnalytics with data from specified projects only.
        """
        analytics = CrossProjectAnalytics()
        projects = self._load_projects_from_registry()

        # Filter to requested project IDs
        requested_projects = [
            p for p in projects if p["id"] in project_ids
        ]

        for project in requested_projects:
            analytics.total_projects_scanned += 1
            snapshot = self._scan_project(project)
            analytics.project_snapshots[project["id"]] = snapshot

            if snapshot.accessible:
                analytics.total_projects_accessible += 1
                analytics.total_tasks_across_projects += snapshot.total_tasks
                analytics.total_steps_across_projects += snapshot.total_steps
                analytics.total_decisions_across_projects += (
                    snapshot.total_decisions
                )

        analytics.scanned_at = datetime.now(timezone.utc)
        return analytics

    def save_analytics(self, analytics: CrossProjectAnalytics,
                      filename: str = "cross-project-analytics.json") -> Path:
        """Save aggregated analytics to global storage.

        Args:
            analytics: The CrossProjectAnalytics object to save.
            filename: Name of the output file.

        Returns:
            Path to the written file.
        """
        output_path = self.global_storage_path / filename
        data = analytics.to_dict()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
            fp.write("\n")

        logger.info("Saved cross-project analytics to %s", output_path)
        return output_path

    def load_analytics(self, filename: str = "cross-project-analytics.json"
                      ) -> Optional[CrossProjectAnalytics]:
        """Load previously saved cross-project analytics.

        Args:
            filename: Name of the analytics file to load.

        Returns:
            CrossProjectAnalytics or None if file does not exist.
        """
        input_path = self.global_storage_path / filename
        if not input_path.is_file():
            logger.warning("Analytics file not found at %s", input_path)
            return None

        try:
            with open(input_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            analytics = CrossProjectAnalytics()
            analytics.total_projects_scanned = data.get(
                "total_projects_scanned", 0
            )
            analytics.total_projects_accessible = data.get(
                "total_projects_accessible", 0
            )
            analytics.total_tasks_across_projects = data.get(
                "total_tasks_across_projects", 0
            )
            analytics.total_steps_across_projects = data.get(
                "total_steps_across_projects", 0
            )
            analytics.total_decisions_across_projects = data.get(
                "total_decisions_across_projects", 0
            )
            analytics.global_top_files = data.get("global_top_files", {})
            analytics.global_error_patterns = data.get(
                "global_error_patterns", {}
            )

            if data.get("scanned_at"):
                analytics.scanned_at = datetime.fromisoformat(data["scanned_at"])

            return analytics
        except Exception as e:
            logger.warning("Failed to load analytics from %s: %s", input_path, e)
            return None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _resolve_hub_registry(self, provided_path: Optional[str]) -> Path:
        """Resolve the hub registry path.

        Searches in this order:
        1. Provided path (if given)
        2. .agent-forge/hub/registry.json relative to current directory
        3. .agent-forge/hub/registry.json relative to detected project root
        """
        if provided_path is not None:
            return Path(provided_path).resolve()

        # Check current directory first
        cwd_registry = Path.cwd() / ".agent-forge" / "hub" / "registry.json"
        if cwd_registry.is_file():
            return cwd_registry

        # Check home directory
        home_registry = Path.home() / ".agent-forge" / "hub" / "registry.json"
        if home_registry.is_file():
            return home_registry

        # Return default (may not exist yet)
        return Path.home() / ".agent-forge" / "hub" / "registry.json"

    def _resolve_global_storage(self, provided_path: Optional[str]) -> Path:
        """Resolve the global storage path.

        Uses ~/.claude-memory-global/ by default, or the provided path.
        """
        if provided_path is not None:
            return Path(provided_path).resolve()

        return Path.home() / GLOBAL_STORAGE_DIR

    def _ensure_global_storage(self) -> None:
        """Create global storage directory if it doesn't exist."""
        self.global_storage_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Global storage directory: %s", self.global_storage_path)

    def _load_projects_from_registry(self) -> list[dict]:
        """Load project list from the hub registry.

        Returns:
            List of project dictionaries. Empty list if registry not found.
        """
        if not self.hub_registry_path.is_file():
            logger.warning("Hub registry not found at %s",
                          self.hub_registry_path)
            return []

        try:
            with open(self.hub_registry_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            projects = data.get("projects", [])
            logger.info("Loaded %d projects from registry", len(projects))
            return projects
        except Exception as e:
            logger.warning("Failed to load hub registry: %s", e)
            return []

    def _scan_project(self, project: dict) -> ProjectMemorySnapshot:
        """Scan a single project's memory storage.

        Args:
            project: Project dictionary from registry.

        Returns:
            ProjectMemorySnapshot with analysis results.
        """
        snapshot = ProjectMemorySnapshot(
            project["id"],
            project.get("name", project["id"]),
            project.get("local_path", "unknown"),
        )

        # Check if project path is accessible
        project_path = Path(snapshot.local_path)
        if not project_path.is_dir():
            snapshot.accessible = False
            snapshot.scan_error = f"Project path not accessible: {snapshot.local_path}"
            logger.warning("Project path not accessible: %s", snapshot.local_path)
            return snapshot

        # Try to load memory storage
        try:
            memory_storage = MemoryStore(str(project_path / ".claude-memory"))
            task_ids = memory_storage.list_tasks()
            snapshot.total_tasks = len(task_ids)

            if snapshot.total_tasks > 0:
                # Compute analytics for this project
                analytics = MemoryAnalytics(str(memory_storage.storage_root))
                analytics.analyze()

                summary = analytics.get_summary()
                snapshot.total_steps = int(
                    summary.get("avg_steps_per_ticket", 0) * snapshot.total_tasks
                )
                snapshot.total_decisions = summary.get("total_decisions", 0)
                snapshot.avg_steps_per_task = summary.get(
                    "avg_steps_per_ticket", 0
                )

                # Extract top files
                snapshot.top_files = [
                    f.to_dict() for f in
                    analytics._file_frequencies.values()
                ][:10]

                # Extract error patterns
                snapshot.error_patterns = analytics.get_error_patterns()

            snapshot.accessible = True
            snapshot.last_scanned = datetime.now(timezone.utc)
            logger.info(
                "Successfully scanned project %s: %d tasks",
                snapshot.project_id,
                snapshot.total_tasks,
            )

        except Exception as e:
            snapshot.accessible = False
            snapshot.scan_error = str(e)
            logger.warning("Failed to scan project %s: %s",
                          snapshot.project_id, e)

        return snapshot

    @property
    def global_storage_root(self) -> Path:
        """The global storage directory used for cross-project data."""
        return self.global_storage_path
