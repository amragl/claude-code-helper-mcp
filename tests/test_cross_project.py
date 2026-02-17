"""Tests for CrossProjectMemory -- cross-project memory aggregation.

All tests use real file storage, real registries, and real TaskMemory objects
with no mocks or placeholders.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import tempfile

import pytest

from claude_code_helper_mcp.analytics.cross_project import (
    CrossProjectMemory,
    CrossProjectAnalytics,
    ProjectMemorySnapshot,
)
from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.models.records import FileAction
from claude_code_helper_mcp.storage.store import MemoryStore


@pytest.fixture
def tmp_global_storage(tmp_path: Path) -> Path:
    """Create a temporary global storage directory."""
    global_dir = tmp_path / ".claude-memory-global"
    global_dir.mkdir(parents=True, exist_ok=True)
    return global_dir


@pytest.fixture
def sample_registry(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Create a sample hub registry with test projects."""
    projects = []

    # Create project 1: CMH with memory
    project1_path = tmp_path / "project1"
    project1_path.mkdir(parents=True, exist_ok=True)
    memory1_path = project1_path / ".claude-memory"
    memory1_path.mkdir(parents=True, exist_ok=True)

    # Create some tasks for project1
    store1 = MemoryStore(str(memory1_path))
    task1 = TaskMemory(ticket_id="CMH-001", title="Task 1", phase="phase-1")
    task1.add_step("Step 1", "First step", "Read", "Success")
    task1.add_step("Step 2", "Second step", "Write", "Created file")
    task1.record_file("src/main.py", FileAction.CREATED, "Created main")
    task1.record_file("src/utils.py", FileAction.CREATED, "Created utils")
    task1.add_decision("Use JSON", "Simple format", ["SQLite", "YAML"])
    store1.save_task(task1)

    task2 = TaskMemory(ticket_id="CMH-002", title="Task 2", phase="phase-1")
    task2.add_step("Analyze", "Analysis step", "Read", "Done")
    task2.record_file("src/main.py", FileAction.MODIFIED, "Updated main")
    task2.add_decision("Add logging", "Debug visibility", ["Silent", "Verbose"])
    store1.save_task(task2)

    projects.append({
        "id": "cmh-project",
        "name": "Claude Code Helper",
        "local_path": str(project1_path),
        "status": "active",
        "health": "healthy",
    })

    # Create project 2: SNOW with memory
    project2_path = tmp_path / "project2"
    project2_path.mkdir(parents=True, exist_ok=True)
    memory2_path = project2_path / ".claude-memory"
    memory2_path.mkdir(parents=True, exist_ok=True)

    store2 = MemoryStore(str(memory2_path))
    task3 = TaskMemory(ticket_id="SNOW-001", title="Discovery Task", phase="phase-1")
    task3.add_step("Query API", "REST call", "Read", "Success")
    task3.add_step("Parse response", "Data extraction", "Write", "Parsed")
    task3.add_step("Store results", "Data persistence", "Write", "Stored")
    task3.record_file("src/discovery.py", FileAction.CREATED, "Discovery module")
    task3.record_file("src/client.py", FileAction.CREATED, "Client module")
    task3.record_file("tests/test_discovery.py", FileAction.CREATED, "Tests")
    task3.add_decision("Use async", "Non-blocking I/O", ["Sync", "Threaded"])
    store2.save_task(task3)

    task4 = TaskMemory(ticket_id="SNOW-002", title="Asset Management", phase="phase-1")
    task4.add_step("Load config", "Configuration", "Read", "Loaded")
    task4.add_step("Connect to SNOW", "API connection", "Read", "Connected")
    task4.record_file("src/client.py", FileAction.MODIFIED, "Added auth")
    task4.record_file("src/config.py", FileAction.CREATED, "Configuration")
    task4.add_decision("Cache credentials", "Performance", ["Always fetch", "Store"])
    store2.save_task(task4)

    projects.append({
        "id": "snow-project",
        "name": "ServiceNow Project",
        "local_path": str(project2_path),
        "status": "active",
        "health": "healthy",
    })

    # Create project 3: Inaccessible project (path doesn't exist)
    projects.append({
        "id": "missing-project",
        "name": "Missing Project",
        "local_path": str(tmp_path / "nonexistent"),
        "status": "active",
        "health": "unhealthy",
    })

    # Write registry to file
    registry_path = tmp_path / "registry.json"
    registry_data = {
        "hub_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "projects": projects,
    }

    with open(registry_path, "w") as f:
        json.dump(registry_data, f, indent=2)

    return registry_path, projects


class TestCrossProjectMemoryInit:
    """Test CrossProjectMemory initialization."""

    def test_init_with_explicit_paths(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "registry.json"
        global_storage = tmp_path / ".claude-memory-global"

        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(global_storage),
        )

        assert cross_project.hub_registry_path == registry_path
        assert cross_project.global_storage_path == global_storage
        assert global_storage.is_dir()

    def test_global_storage_directory_created(self, tmp_path: Path) -> None:
        global_storage = tmp_path / ".claude-memory-global"
        assert not global_storage.exists()

        CrossProjectMemory(global_storage_path=str(global_storage))

        assert global_storage.is_dir()

    def test_registry_path_resolution(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "custom_registry.json"
        cross_project = CrossProjectMemory(hub_registry_path=str(registry_path))
        assert cross_project.hub_registry_path == registry_path


class TestScanSingleProject:
    """Test scanning a single project."""

    def test_scan_accessible_project(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Scan the first accessible project
        project = projects[0]
        snapshot = cross_project._scan_project(project)

        assert snapshot.project_id == "cmh-project"
        assert snapshot.project_name == "Claude Code Helper"
        assert snapshot.accessible is True
        assert snapshot.total_tasks == 2
        assert snapshot.total_decisions > 0
        assert snapshot.last_scanned is not None
        assert snapshot.scan_error is None

    def test_scan_inaccessible_project(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Scan the inaccessible project
        project = projects[2]
        snapshot = cross_project._scan_project(project)

        assert snapshot.project_id == "missing-project"
        assert snapshot.accessible is False
        assert snapshot.scan_error is not None
        assert snapshot.total_tasks == 0


class TestScanAllProjects:
    """Test scanning all projects."""

    def test_scan_all_projects(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()

        assert analytics.total_projects_scanned == 3
        assert analytics.total_projects_accessible == 2
        assert analytics.total_tasks_across_projects == 4
        assert analytics.scanned_at is not None
        assert len(analytics.project_snapshots) == 3

    def test_aggregated_statistics(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()

        # Check aggregated data
        assert analytics.total_tasks_across_projects == 4
        assert analytics.total_decisions_across_projects > 0
        assert analytics.total_steps_across_projects > 0

    def test_global_top_files(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()

        # Both projects modify src/main.py and src/config.py
        assert len(analytics.global_top_files) > 0
        assert analytics.global_top_files.get("src/main.py", 0) > 1

    def test_error_patterns_aggregation(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()

        # Error patterns should be aggregated (even if empty)
        assert isinstance(analytics.global_error_patterns, dict)


class TestScanSpecificProjects:
    """Test scanning specific projects only."""

    def test_scan_specific_projects(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Scan only the first project
        analytics = cross_project.scan_projects(["cmh-project"])

        assert analytics.total_projects_scanned == 1
        assert analytics.total_projects_accessible == 1
        assert analytics.total_tasks_across_projects == 2
        assert "cmh-project" in analytics.project_snapshots

    def test_scan_multiple_specific_projects(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Scan specific projects
        analytics = cross_project.scan_projects(["cmh-project", "snow-project"])

        assert analytics.total_projects_scanned == 2
        assert analytics.total_projects_accessible == 2
        assert analytics.total_tasks_across_projects == 4

    def test_scan_nonexistent_project(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Request a project that doesn't exist in the registry
        analytics = cross_project.scan_projects(["nonexistent-id"])

        assert analytics.total_projects_scanned == 0


class TestSaveAndLoadAnalytics:
    """Test saving and loading analytics."""

    def test_save_analytics(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()
        output_path = cross_project.save_analytics(analytics)

        assert output_path.is_file()
        assert output_path.name == "cross-project-analytics.json"
        assert tmp_global_storage in output_path.parents

    def test_save_with_custom_filename(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()
        output_path = cross_project.save_analytics(
            analytics,
            filename="custom-report.json",
        )

        assert output_path.name == "custom-report.json"
        assert output_path.is_file()

    def test_load_analytics(self, sample_registry: tuple, tmp_global_storage: Path) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Save first
        analytics_saved = cross_project.scan_all_projects()
        cross_project.save_analytics(analytics_saved)

        # Load it back
        analytics_loaded = cross_project.load_analytics()

        assert analytics_loaded is not None
        assert analytics_loaded.total_projects_scanned == 3
        assert analytics_loaded.total_projects_accessible == 2
        assert analytics_loaded.total_tasks_across_projects == 4

    def test_load_nonexistent_analytics(self, tmp_path: Path, tmp_global_storage: Path) -> None:
        registry_path = tmp_path / "registry.json"
        registry_data = {
            "hub_version": "1.0.0",
            "projects": [],
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.load_analytics()
        assert analytics is None


class TestProjectMemorySnapshot:
    """Test ProjectMemorySnapshot serialization."""

    def test_snapshot_to_dict(self) -> None:
        snapshot = ProjectMemorySnapshot(
            project_id="test-project",
            project_name="Test Project",
            local_path="/path/to/project",
        )
        snapshot.total_tasks = 5
        snapshot.total_steps = 25
        snapshot.total_decisions = 10
        snapshot.accessible = True
        snapshot.last_scanned = datetime.now(timezone.utc)

        data = snapshot.to_dict()

        assert data["project_id"] == "test-project"
        assert data["project_name"] == "Test Project"
        assert data["total_tasks"] == 5
        assert data["total_steps"] == 25
        assert data["total_decisions"] == 10
        assert data["accessible"] is True
        assert data["last_scanned"] is not None


class TestCrossProjectAnalytics:
    """Test CrossProjectAnalytics serialization."""

    def test_analytics_to_dict(self) -> None:
        analytics = CrossProjectAnalytics()
        analytics.total_projects_scanned = 3
        analytics.total_projects_accessible = 2
        analytics.total_tasks_across_projects = 15
        analytics.total_steps_across_projects = 75
        analytics.total_decisions_across_projects = 30
        analytics.scanned_at = datetime.now(timezone.utc)

        data = analytics.to_dict()

        assert data["total_projects_scanned"] == 3
        assert data["total_projects_accessible"] == 2
        assert data["total_tasks_across_projects"] == 15
        assert data["scanned_at"] is not None

    def test_analytics_with_project_snapshots(self) -> None:
        analytics = CrossProjectAnalytics()
        snapshot = ProjectMemorySnapshot(
            "proj1",
            "Project 1",
            "/path/to/proj1",
        )
        snapshot.accessible = True
        snapshot.total_tasks = 5

        analytics.project_snapshots["proj1"] = snapshot
        data = analytics.to_dict()

        assert "proj1" in data["project_snapshots"]
        assert data["project_snapshots"]["proj1"]["total_tasks"] == 5


class TestLoadProjectsFromRegistry:
    """Test loading projects from the hub registry."""

    def test_load_projects_valid_registry(self, sample_registry: tuple) -> None:
        registry_path, projects = sample_registry
        cross_project = CrossProjectMemory(hub_registry_path=str(registry_path))

        loaded_projects = cross_project._load_projects_from_registry()

        assert len(loaded_projects) == 3
        assert loaded_projects[0]["id"] == "cmh-project"
        assert loaded_projects[1]["id"] == "snow-project"

    def test_load_projects_missing_registry(self, tmp_path: Path) -> None:
        missing_registry = tmp_path / "missing" / "registry.json"
        cross_project = CrossProjectMemory(hub_registry_path=str(missing_registry))

        loaded_projects = cross_project._load_projects_from_registry()

        assert loaded_projects == []

    def test_load_projects_malformed_json(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "malformed.json"
        registry_path.write_text("{ invalid json }")

        cross_project = CrossProjectMemory(hub_registry_path=str(registry_path))
        loaded_projects = cross_project._load_projects_from_registry()

        assert loaded_projects == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_registry(self, tmp_path: Path, tmp_global_storage: Path) -> None:
        registry_path = tmp_path / "empty_registry.json"
        registry_data = {
            "hub_version": "1.0.0",
            "projects": [],
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()

        assert analytics.total_projects_scanned == 0
        assert analytics.total_projects_accessible == 0

    def test_project_with_empty_memory(self, tmp_path: Path, tmp_global_storage: Path) -> None:
        # Create a project with empty .claude-memory/
        project_path = tmp_path / "empty_project"
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / ".claude-memory").mkdir(parents=True, exist_ok=True)

        registry_path = tmp_path / "registry.json"
        registry_data = {
            "hub_version": "1.0.0",
            "projects": [{
                "id": "empty-project",
                "name": "Empty Project",
                "local_path": str(project_path),
            }],
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        analytics = cross_project.scan_all_projects()

        assert analytics.total_projects_scanned == 1
        assert analytics.total_projects_accessible == 1
        assert analytics.total_tasks_across_projects == 0

    def test_project_with_corrupted_memory(self, tmp_path: Path, tmp_global_storage: Path) -> None:
        # Create a project with corrupted memory data
        project_path = tmp_path / "corrupted_project"
        project_path.mkdir(parents=True, exist_ok=True)
        memory_path = project_path / ".claude-memory"
        memory_path.mkdir(parents=True, exist_ok=True)

        # Write corrupted task file
        tasks_dir = memory_path / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "task-CMH-001.json").write_text("{ corrupted json }")

        registry_path = tmp_path / "registry.json"
        registry_data = {
            "hub_version": "1.0.0",
            "projects": [{
                "id": "corrupted-project",
                "name": "Corrupted Project",
                "local_path": str(project_path),
            }],
        }
        with open(registry_path, "w") as f:
            json.dump(registry_data, f)

        cross_project = CrossProjectMemory(
            hub_registry_path=str(registry_path),
            global_storage_path=str(tmp_global_storage),
        )

        # Should handle gracefully without crashing
        analytics = cross_project.scan_all_projects()
        assert analytics.total_projects_scanned == 1
        # The project might still be marked as accessible even with
        # corrupted data, as long as the directory structure is OK
