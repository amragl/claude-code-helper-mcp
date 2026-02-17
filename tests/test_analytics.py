"""Tests for MemoryAnalytics -- pattern analysis across tasks.

All tests use real file storage and real TaskMemory objects with no mocks.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_code_helper_mcp.analytics.analytics import MemoryAnalytics
from claude_code_helper_mcp.models.records import FileAction
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore


@pytest.fixture()
def tmp_store(tmp_path: Path) -> MemoryStore:
    """Return a MemoryStore rooted in a fresh temp directory."""
    return MemoryStore(str(tmp_path / ".claude-memory"))


@pytest.fixture()
def sample_tasks(tmp_store: MemoryStore) -> list[TaskMemory]:
    """Create and save multiple sample tasks for analysis."""
    tasks = []

    # Task 1: Simple task with few steps
    task1 = TaskMemory(ticket_id="CMH-001", title="Task 1", phase="phase-1")
    task1.add_step("Step 1", "First step", "Read", "Success")
    task1.add_step("Step 2", "Second step", "Write", "Created file")
    task1.record_file("src/main.py", FileAction.CREATED, "Created main entry point")
    task1.record_file("src/utils.py", FileAction.CREATED, "Created utils module")
    task1.add_decision("Use JSON storage", "Simple and fast", ["SQLite", "TOML"])
    tmp_store.save_task(task1)
    tasks.append(task1)

    # Task 2: More complex task
    task2 = TaskMemory(ticket_id="CMH-002", title="Task 2", phase="phase-1")
    task2.add_step("Analyze requirements", "Requirements analysis", "Read", "Done")
    task2.add_step("Design schema", "Schema design", "Write", "Designed")
    task2.add_step("Implement models", "Implementation", "Write", "Created models.py")
    task2.add_step("Write tests", "Testing", "Write", "3 tests pass")
    task2.record_file("src/main.py", FileAction.MODIFIED, "Updated main entry")
    task2.record_file("src/models.py", FileAction.CREATED, "Created data models")
    task2.record_file("tests/test_models.py", FileAction.CREATED, "Added model tests")
    task2.add_decision("Use Pydantic", "Type validation", ["dataclasses", "attrs"])
    task2.add_decision("Store as JSON", "Human readable", ["YAML", "Pickle"])
    tmp_store.save_task(task2)
    tasks.append(task2)

    # Task 3: Failed task with errors
    task3 = TaskMemory(ticket_id="CMH-003", title="Task 3", phase="phase-2")
    task3.add_step("Step 1", "Attempt 1", "Write", "Unexpected error", success=False)
    task3.add_step("Step 2", "Attempt 2", "Write", "Connection timeout", success=False)
    task3.add_step("Step 3", "Attempt 3", "Write", "Success on retry", success=True)
    task3.record_file("src/main.py", FileAction.MODIFIED, "Fixed bug")
    task3.record_file("src/config.py", FileAction.CREATED, "Added config")
    task3.record_file("tests/test_integration.py", FileAction.CREATED, "Added tests")
    task3.status = TaskStatus.COMPLETED
    task3.completed_at = datetime.now(timezone.utc)
    tmp_store.save_task(task3)
    tasks.append(task3)

    # Task 4: Another task using same files
    task4 = TaskMemory(ticket_id="CMH-004", title="Task 4", phase="phase-2")
    task4.add_step("Update main", "Code update", "Write", "Done")
    task4.add_step("Update config", "Config update", "Write", "Done")
    task4.record_file("src/main.py", FileAction.MODIFIED, "Added feature")
    task4.record_file("src/config.py", FileAction.MODIFIED, "Updated config")
    task4.add_decision("Add caching", "Performance", ["No caching", "Redis"])
    tmp_store.save_task(task4)
    tasks.append(task4)

    return tasks


class TestMemoryAnalyticsInit:
    """Test MemoryAnalytics initialization."""

    def test_init_with_explicit_path(self, tmp_path: Path) -> None:
        storage_path = str(tmp_path / ".claude-memory")
        analytics = MemoryAnalytics(storage_path)
        assert analytics.store.storage_root.is_dir()

    def test_init_auto_detects_storage(self, tmp_store: MemoryStore) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        assert analytics.store.storage_root == tmp_store.storage_root


class TestAnalyzeFunction:
    """Test the analyze() method."""

    def test_analyze_empty_storage(self, tmp_path: Path) -> None:
        storage_path = str(tmp_path / ".claude-memory")
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        assert analytics._analyzed is True
        assert len(analytics._task_patterns) == 0

    def test_analyze_single_task(self, tmp_store: MemoryStore) -> None:
        task = TaskMemory(ticket_id="CMH-100", title="Test", phase="phase-1")
        task.add_step("Action 1", "First action")
        task.add_step("Action 2", "Second action")
        task.record_file("src/test.py", FileAction.CREATED, "Test file")
        tmp_store.save_task(task)

        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        assert len(analytics._task_patterns) == 1
        pattern = analytics._task_patterns["CMH-100"]
        assert pattern.step_count == 2
        assert pattern.files_modified == ["src/test.py"]

    def test_analyze_multiple_tasks(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        assert len(analytics._task_patterns) == 4
        assert "CMH-001" in analytics._task_patterns
        assert "CMH-002" in analytics._task_patterns

    def test_analyze_filters_by_since(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        # Set task 1 and 2 to old dates
        storage_path = str(tmp_store.storage_root)

        # Create tasks with specific dates
        old_task = TaskMemory(ticket_id="CMH-OLD", title="Old", phase="phase-1")
        old_task.started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        tmp_store.save_task(old_task)

        new_task = TaskMemory(ticket_id="CMH-NEW", title="New", phase="phase-1")
        new_task.started_at = datetime(2026, 2, 15, tzinfo=timezone.utc)
        tmp_store.save_task(new_task)

        # Analyze since 2026-02-01
        analytics = MemoryAnalytics(storage_path)
        since = datetime(2026, 2, 1, tzinfo=timezone.utc)
        analytics.analyze(since=since)

        assert "CMH-NEW" in analytics._task_patterns
        assert "CMH-OLD" not in analytics._task_patterns


class TestTaskPatternAnalysis:
    """Test analysis of individual task patterns."""

    def test_step_count_aggregation(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        pattern1 = analytics._task_patterns["CMH-001"]
        assert pattern1.step_count == 2

        pattern2 = analytics._task_patterns["CMH-002"]
        assert pattern2.step_count == 4

        pattern3 = analytics._task_patterns["CMH-003"]
        assert pattern3.step_count == 3

    def test_file_modification_tracking(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        # src/main.py should be modified in 4 tasks
        assert "src/main.py" in analytics._file_frequencies
        freq = analytics._file_frequencies["src/main.py"]
        assert freq.modification_count == 4
        assert set(freq.tasks_that_modified) >= {"CMH-001", "CMH-002", "CMH-003", "CMH-004"}

    def test_decision_counting(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        # Total decisions across tasks (1 + 2 + 0 + 1 = 4)
        assert analytics._decision_patterns.total_decisions == 4

    def test_error_tracking(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        pattern = analytics._task_patterns["CMH-003"]
        # Task 3 has 2 failed steps
        assert len(pattern.errors) == 2


class TestGetSummary:
    """Test the get_summary() method."""

    def test_summary_without_analyze_raises(self, tmp_path: Path) -> None:
        storage_path = str(tmp_path / ".claude-memory")
        analytics = MemoryAnalytics(storage_path)
        with pytest.raises(RuntimeError):
            analytics.get_summary()

    def test_summary_empty_storage(self, tmp_path: Path) -> None:
        storage_path = str(tmp_path / ".claude-memory")
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        summary = analytics.get_summary()

        assert summary["total_tasks_analyzed"] == 0
        assert summary["avg_steps_per_ticket"] == 0

    def test_summary_with_tasks(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        summary = analytics.get_summary()

        assert summary["total_tasks_analyzed"] == 4
        assert summary["avg_steps_per_ticket"] > 0
        assert summary["total_files_modified"] > 0
        assert summary["total_decisions"] == 4

    def test_summary_includes_top_files(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        summary = analytics.get_summary()

        top_files = summary["top_files"]
        assert len(top_files) > 0
        assert top_files[0]["path"] == "src/main.py"
        assert top_files[0]["modification_count"] >= 3


class TestFileModificationHeatMap:
    """Test the get_file_modification_heat_map() method."""

    def test_heat_map_without_analyze_raises(self, tmp_path: Path) -> None:
        storage_path = str(tmp_path / ".claude-memory")
        analytics = MemoryAnalytics(storage_path)
        with pytest.raises(RuntimeError):
            analytics.get_file_modification_heat_map()

    def test_heat_map_structure(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        heat_map = analytics.get_file_modification_heat_map()

        assert isinstance(heat_map, dict)
        assert all(isinstance(k, str) and isinstance(v, int) for k, v in heat_map.items())
        assert heat_map["src/main.py"] == 4


class TestErrorPatterns:
    """Test the get_error_patterns() method."""

    def test_error_patterns_with_failures(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        errors = analytics.get_error_patterns()

        assert isinstance(errors, dict)
        # The fixture has failed steps with descriptions "Attempt 1" and "Attempt 2"
        assert "Attempt 1" in errors
        assert "Attempt 2" in errors

    def test_error_patterns_counts(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        errors = analytics.get_error_patterns()

        assert errors["Attempt 1"] == 1
        assert errors["Attempt 2"] == 1


class TestDecisionPatterns:
    """Test decision pattern analysis."""

    def test_decision_types_aggregation(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        decision_types = analytics.get_top_decision_types()

        assert isinstance(decision_types, dict)
        assert len(decision_types) > 0

    def test_alternatives_averaging(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()

        # Each decision has 2 alternatives
        assert analytics._decision_patterns.avg_alternatives_per_decision > 0


class TestJsonSerialization:
    """Test JSON serialization of analysis results."""

    def test_to_json_dict_without_analyze_raises(self, tmp_path: Path) -> None:
        storage_path = str(tmp_path / ".claude-memory")
        analytics = MemoryAnalytics(storage_path)
        with pytest.raises(RuntimeError):
            analytics.to_json_dict()

    def test_to_json_dict_structure(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        json_data = analytics.to_json_dict()

        assert "summary" in json_data
        assert "detailed" in json_data
        assert "file_heat_map" in json_data
        assert "error_patterns" in json_data
        assert "decision_types" in json_data
        assert "analyzed_at" in json_data

    def test_json_is_serializable(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        import json

        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        json_data = analytics.to_json_dict()

        # Should be able to serialize to JSON without errors
        json_str = json.dumps(json_data)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["summary"]["total_tasks_analyzed"] == 4


class TestAllPatterns:
    """Test the get_all_patterns() method."""

    def test_all_patterns_structure(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        patterns = analytics.get_all_patterns()

        assert "tasks" in patterns
        assert "files" in patterns
        assert "decisions" in patterns
        assert len(patterns["tasks"]) == 4

    def test_task_patterns_detail(
        self, tmp_store: MemoryStore, sample_tasks: list[TaskMemory]
    ) -> None:
        storage_path = str(tmp_store.storage_root)
        analytics = MemoryAnalytics(storage_path)
        analytics.analyze()
        patterns = analytics.get_all_patterns()

        task_pattern = patterns["tasks"]["CMH-002"]
        assert task_pattern["ticket_id"] == "CMH-002"
        assert task_pattern["step_count"] == 4
        assert task_pattern["decisions_count"] == 2
        assert "duration_seconds" in task_pattern
