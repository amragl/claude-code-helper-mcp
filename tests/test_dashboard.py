"""Tests for DeveloperDashboard -- comprehensive memory dashboard visualization.

All tests use real file storage and real TaskMemory objects with no mocks.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_code_helper_mcp.analytics.dashboard import (
    DecisionTreeNode,
    DeveloperDashboard,
    FileHeatMapEntry,
    InterventionSummary,
    TaskTimelineEntry,
    WindowStateView,
)
from claude_code_helper_mcp.models.records import DecisionRecord, FileAction
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.models.window import MemoryWindow
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


@pytest.fixture()
def tmp_store(tmp_path: Path) -> MemoryStore:
    """Return a MemoryStore rooted in a fresh temp directory."""
    return MemoryStore(str(tmp_path / ".claude-memory"))


@pytest.fixture()
def tmp_window_manager(tmp_path: Path) -> WindowManager:
    """Return a WindowManager rooted in a fresh temp directory."""
    return WindowManager(str(tmp_path / ".claude-memory"))


@pytest.fixture()
def sample_tasks(tmp_store: MemoryStore) -> list[TaskMemory]:
    """Create and save multiple sample tasks for dashboard testing."""
    tasks = []

    # Task 1: Simple completed task
    now = datetime.now(timezone.utc)
    task1 = TaskMemory(
        ticket_id="CMH-001",
        title="Foundation setup",
        phase="phase-1",
        status=TaskStatus.COMPLETED,
        started_at=now - timedelta(hours=2),
        completed_at=now - timedelta(hours=1),
    )
    task1.add_step("Initialize project", "Setup", "Write", "Done")
    task1.add_step("Create structure", "Setup", "Write", "Created dirs")
    task1.record_file("src/main.py", FileAction.CREATED, "Main entry")
    task1.record_file("src/config.py", FileAction.CREATED, "Config module")
    task1.add_decision("Use Click", "Simple CLI framework", ["argparse", "typer"])
    task1.add_decision("JSON storage", "Human readable", ["YAML", "SQLite"])
    tmp_store.save_task(task1)
    tasks.append(task1)

    # Task 2: Active task with multiple files
    task2 = TaskMemory(
        ticket_id="CMH-002",
        title="MCP tools implementation",
        phase="phase-2",
        status=TaskStatus.ACTIVE,
        started_at=now - timedelta(minutes=30),
    )
    task2.add_step("Design MCP interface", "Design", "Read", "Done")
    task2.add_step("Implement record_step", "Implement", "Write", "Created")
    task2.add_step("Add tests", "Test", "Write", "10 tests pass")
    task2.record_file("src/mcp/server.py", FileAction.CREATED, "MCP server")
    task2.record_file("src/mcp/tools.py", FileAction.CREATED, "Tool implementations")
    task2.record_file("src/main.py", FileAction.MODIFIED, "Integrated MCP")
    task2.record_file("tests/test_mcp.py", FileAction.CREATED, "MCP tests")
    task2.add_decision("Async MCP", "Better performance", ["Sync", "Hybrid"])
    task2.add_decision("Tool namespacing", "Organization", ["Flat", "Tree"])
    tmp_store.save_task(task2)
    tasks.append(task2)

    # Task 3: Failed task with many retries
    task3 = TaskMemory(
        ticket_id="CMH-003",
        title="Drift detection",
        phase="phase-5",
        status=TaskStatus.COMPLETED,
        started_at=now - timedelta(hours=4),
        completed_at=now - timedelta(hours=3),
    )
    task3.add_step("Analyze requirements", "Planning", "Read", "Done")
    task3.add_step("Attempt 1", "Implementation", "Write", "Test failed", success=False)
    task3.add_step("Attempt 2", "Debug", "Write", "Edge case found", success=False)
    task3.add_step("Attempt 3", "Fix", "Write", "Tests pass", success=True)
    task3.record_file("src/detection/drift.py", FileAction.CREATED, "Drift detector")
    task3.record_file("src/main.py", FileAction.MODIFIED, "Integrated detector")
    task3.record_file("tests/test_drift.py", FileAction.CREATED, "Drift tests")
    task3.add_decision("Scope analysis method", "Determines drift", ["File-based", "Action-based"])
    task3.metadata["interventions"] = {
        "drift_report": {
            "severity": "high",
            "summary": "Task drift detected in file scope",
        }
    }
    tmp_store.save_task(task3)
    tasks.append(task3)

    # Task 4: Task with many file modifications (heat map test)
    task4 = TaskMemory(
        ticket_id="CMH-004",
        title="Error loop detection",
        phase="phase-5",
        status=TaskStatus.COMPLETED,
        started_at=now - timedelta(hours=3),
        completed_at=now - timedelta(hours=2),
    )
    for i in range(5):
        task4.add_step(f"Step {i+1}", "Development", "Write", "Progress")
    task4.record_file("src/main.py", FileAction.MODIFIED, "Updated")
    task4.record_file("src/detection/errors.py", FileAction.CREATED, "Error detector")
    task4.record_file("src/detection/errors.py", FileAction.MODIFIED, "Enhanced")
    task4.record_file("tests/test_errors.py", FileAction.CREATED, "Error tests")
    task4.record_file("tests/test_errors.py", FileAction.MODIFIED, "More tests")
    task4.add_decision("Threshold value", "Error detection threshold", ["2", "3", "5"])
    tmp_store.save_task(task4)
    tasks.append(task4)

    return tasks


@pytest.fixture()
def initialized_window(tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> MemoryWindow:
    """Create an initialized window with sample tasks."""
    window = MemoryWindow(window_size=3)

    # Add completed tasks to window (as TaskMemory objects, not IDs)
    window.completed_tasks = [sample_tasks[0], sample_tasks[2], sample_tasks[3]]

    # Set current task
    window.current_task = sample_tasks[1]

    # Save the window using the store
    tmp_store.save_window(window)

    return window


class TestTaskTimelineEntry:
    """Test TaskTimelineEntry model."""

    def test_create_entry(self) -> None:
        """Test creating a timeline entry."""
        started = datetime.now(timezone.utc) - timedelta(hours=1)
        completed = datetime.now(timezone.utc)

        entry = TaskTimelineEntry(
            ticket_id="CMH-001",
            title="Test Task",
            started_at=started,
            completed_at=completed,
            status="completed",
            step_count=5,
        )

        assert entry.ticket_id == "CMH-001"
        assert entry.title == "Test Task"
        assert entry.status == "completed"
        assert entry.step_count == 5
        assert entry.duration_seconds > 0

    def test_entry_to_dict(self) -> None:
        """Test converting entry to dictionary."""
        started = datetime.now(timezone.utc) - timedelta(hours=1)
        completed = datetime.now(timezone.utc)

        entry = TaskTimelineEntry(
            ticket_id="CMH-001",
            title="Test Task",
            started_at=started,
            completed_at=completed,
            status="completed",
            step_count=5,
        )

        data = entry.to_dict()

        assert data["ticket_id"] == "CMH-001"
        assert data["title"] == "Test Task"
        assert data["status"] == "completed"
        assert data["step_count"] == 5
        assert "started_at" in data
        assert "completed_at" in data
        assert data["duration_seconds"] > 0

    def test_active_entry_duration(self) -> None:
        """Test duration calculation for active tasks."""
        started = datetime.now(timezone.utc) - timedelta(minutes=30)

        entry = TaskTimelineEntry(
            ticket_id="CMH-001",
            title="Active Task",
            started_at=started,
            completed_at=None,
            status="active",
            step_count=3,
        )

        assert entry.duration_seconds > 0
        assert entry.duration_seconds < 2000  # Less than 30 minutes in seconds


class TestDecisionTreeNode:
    """Test DecisionTreeNode model."""

    def test_create_decision_node(self) -> None:
        """Test creating a decision tree node."""
        node = DecisionTreeNode(
            task_id="CMH-001",
            decision_text="Use JSON storage",
            reasoning="Simple and human-readable",
            alternatives=["SQLite", "YAML"],
        )

        assert node.task_id == "CMH-001"
        assert node.decision == "Use JSON storage"
        assert node.reasoning == "Simple and human-readable"
        assert len(node.alternatives) == 2

    def test_node_to_dict(self) -> None:
        """Test converting node to dictionary."""
        node = DecisionTreeNode(
            task_id="CMH-001",
            decision_text="Use Click",
            reasoning="Lightweight framework",
            alternatives=["argparse", "typer"],
        )

        data = node.to_dict()

        assert data["task_id"] == "CMH-001"
        assert data["decision"] == "Use Click"
        assert data["reasoning"] == "Lightweight framework"
        assert data["alternatives"] == ["argparse", "typer"]


class TestFileHeatMapEntry:
    """Test FileHeatMapEntry model."""

    def test_heat_score_cold(self) -> None:
        """Test heat score for single modification."""
        entry = FileHeatMapEntry("src/main.py", 1, ["CMH-001"])
        assert entry.heat_score == "cold"

    def test_heat_score_cool(self) -> None:
        """Test heat score for 2-3 modifications."""
        entry = FileHeatMapEntry("src/main.py", 2, ["CMH-001", "CMH-002"])
        assert entry.heat_score == "cool"

    def test_heat_score_warm(self) -> None:
        """Test heat score for 4-6 modifications."""
        entry = FileHeatMapEntry("src/main.py", 5, ["CMH-001", "CMH-002", "CMH-003", "CMH-004", "CMH-005"])
        assert entry.heat_score == "warm"

    def test_heat_score_hot(self) -> None:
        """Test heat score for 7+ modifications."""
        entry = FileHeatMapEntry("src/main.py", 10, ["CMH-" + str(i) for i in range(10)])
        assert entry.heat_score == "hot"

    def test_entry_to_dict(self) -> None:
        """Test converting heat map entry to dictionary."""
        entry = FileHeatMapEntry("src/main.py", 3, ["CMH-001", "CMH-002", "CMH-003"])
        data = entry.to_dict()

        assert data["file_path"] == "src/main.py"
        assert data["modification_count"] == 3
        assert data["heat_score"] == "cool"
        assert len(data["tasks_modified"]) == 3


class TestInterventionSummary:
    """Test InterventionSummary model."""

    def test_create_empty_summary(self) -> None:
        """Test creating an empty intervention summary."""
        summary = InterventionSummary()

        assert len(summary.drift_detections) == 0
        assert len(summary.error_loop_detections) == 0
        assert len(summary.confusion_detections) == 0
        assert len(summary.scope_creep_detections) == 0

    def test_add_detections(self) -> None:
        """Test adding different types of detections."""
        summary = InterventionSummary()

        summary.add_drift("CMH-001", "high", "File scope mismatch")
        summary.add_error_loop("CMH-002", "test_run", 3)
        summary.add_confusion("CMH-003", "file_not_found", "/path/to/missing.py")
        summary.add_scope_creep("CMH-004", "src/unrelated.py", "Out of scope feature")

        assert len(summary.drift_detections) == 1
        assert len(summary.error_loop_detections) == 1
        assert len(summary.confusion_detections) == 1
        assert len(summary.scope_creep_detections) == 1

    def test_summary_to_dict(self) -> None:
        """Test converting summary to dictionary."""
        summary = InterventionSummary()

        summary.add_drift("CMH-001", "high", "File scope mismatch")
        summary.add_error_loop("CMH-002", "test_run", 3)

        data = summary.to_dict()

        assert len(data["drift_detections"]) == 1
        assert len(data["error_loop_detections"]) == 1
        assert data["total_detections"] == 2

    def test_total_detections_count(self) -> None:
        """Test that total_detections is calculated correctly."""
        summary = InterventionSummary()

        summary.add_drift("CMH-001", "high", "Drift")
        summary.add_drift("CMH-002", "medium", "Drift")
        summary.add_error_loop("CMH-003", "action", 3)
        summary.add_confusion("CMH-004", "type", "details")

        data = summary.to_dict()

        assert data["total_detections"] == 4


class TestWindowStateView:
    """Test WindowStateView model."""

    def test_create_window_view(self) -> None:
        """Test creating a window state view."""
        view = WindowStateView(
            window_size=4,
            completed_tasks=["CMH-001", "CMH-002"],
            active_task="CMH-003",
        )

        assert view.window_size == 4
        assert len(view.completed_tasks) == 2
        assert view.active_task == "CMH-003"
        assert view.total_in_window == 3

    def test_window_occupancy(self) -> None:
        """Test window occupancy string."""
        view = WindowStateView(
            window_size=4,
            completed_tasks=["CMH-001", "CMH-002"],
            active_task="CMH-003",
        )

        data = view.to_dict()
        assert data["window_occupancy"] == "3/4"

    def test_window_view_to_dict(self) -> None:
        """Test converting window view to dictionary."""
        view = WindowStateView(
            window_size=4,
            completed_tasks=["CMH-001", "CMH-002"],
            active_task="CMH-003",
        )

        data = view.to_dict()

        assert data["window_size"] == 4
        assert data["total_in_window"] == 3
        assert data["active_task"] == "CMH-003"
        assert len(data["completed_tasks_in_window"]) == 2


class TestDeveloperDashboardGeneration:
    """Test DeveloperDashboard generation and views."""

    def test_dashboard_init(self, tmp_store: MemoryStore) -> None:
        """Test dashboard initialization."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)

        assert dashboard.store is not None
        assert dashboard.window_manager is not None

    def test_dashboard_generate(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test dashboard generation with sample data."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        assert dashboard._generated is True

    def test_timeline_generation(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test task timeline generation."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        timeline = dashboard.get_timeline()

        assert len(timeline) == 4
        # Timeline should be sorted by start time
        assert timeline[0]["ticket_id"] in ["CMH-001", "CMH-003", "CMH-004"]

    def test_decision_tree_generation(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test decision tree generation."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        tree = dashboard.get_decision_tree()

        assert len(tree) > 0
        # Task 1 has 2 decisions, Task 2 has 2, Task 3 has 1, Task 4 has 1
        assert len(tree) == 6

    def test_file_heat_map_generation(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test file modification heat map generation."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        heat_map = dashboard.get_file_heat_map()

        assert len(heat_map) > 0
        # src/main.py should be most modified (appears in tasks 1, 2, 3, 4 = 4 times)
        assert heat_map[0]["file_path"] == "src/main.py"
        assert heat_map[0]["modification_count"] >= 3

    def test_heat_map_sorting(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test that heat map is sorted by modification count (descending)."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        heat_map = dashboard.get_file_heat_map()

        # Verify sorted
        for i in range(len(heat_map) - 1):
            assert heat_map[i]["modification_count"] >= heat_map[i + 1]["modification_count"]

    def test_intervention_summary_generation(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        """Test intervention summary generation."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        interventions = dashboard.get_intervention_summary()

        assert "drift_detections" in interventions
        assert "error_loop_detections" in interventions
        assert "confusion_detections" in interventions
        assert "scope_creep_detections" in interventions
        assert "total_detections" in interventions

    def test_window_state_generation(
        self,
        tmp_store: MemoryStore,
        sample_tasks: list[TaskMemory],
        initialized_window: MemoryWindow,
    ) -> None:
        """Test window state generation."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        window = dashboard.get_window_state()

        assert "window_size" in window
        assert "completed_tasks_in_window" in window
        assert "active_task" in window

    def test_json_serialization(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test complete JSON serialization."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        data = dashboard.to_json_dict()

        assert "generated_at" in data
        assert "timeline" in data
        assert "decision_tree" in data
        assert "file_heat_map" in data
        assert "interventions" in data
        assert "window_state" in data
        assert "summary" in data

    def test_generate_without_call_raises(self, tmp_store: MemoryStore) -> None:
        """Test that accessing views without generate() raises error."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)

        with pytest.raises(RuntimeError, match="Call generate"):
            dashboard.get_timeline()

    def test_empty_dashboard(self, tmp_store: MemoryStore) -> None:
        """Test dashboard generation with no tasks."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        timeline = dashboard.get_timeline()
        tree = dashboard.get_decision_tree()
        heat_map = dashboard.get_file_heat_map()

        assert len(timeline) == 0
        assert len(tree) == 0
        assert len(heat_map) == 0

    def test_summary_statistics(self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]) -> None:
        """Test summary statistics calculation."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        data = dashboard.to_json_dict()
        summary = data["summary"]

        assert summary["total_tasks"] == 4
        assert summary["total_decisions"] > 0
        assert summary["files_tracked"] > 0

    def test_timeline_chronological_order(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        """Test that timeline is in chronological order."""
        dashboard = DeveloperDashboard(tmp_store.storage_root)
        dashboard.generate()

        timeline = dashboard.get_timeline()

        for i in range(len(timeline) - 1):
            current_time = datetime.fromisoformat(timeline[i]["started_at"])
            next_time = datetime.fromisoformat(timeline[i + 1]["started_at"])
            assert current_time <= next_time
