"""Tests for all Pydantic data models -- StepRecord, FileRecord, BranchRecord,
DecisionRecord, TaskMemory, RecoveryContext, MemoryWindow.

All tests use real data, real I/O, and real Pydantic validation.
No mocks, no stubs, no fakes.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from claude_code_helper_mcp.models import (
    BranchAction,
    BranchRecord,
    DecisionRecord,
    FileAction,
    FileRecord,
    MemoryWindow,
    RecoveryContext,
    StepRecord,
    TaskMemory,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# StepRecord Tests
# ---------------------------------------------------------------------------


class TestStepRecord:
    def test_create_minimal(self):
        step = StepRecord(step_number=1, action="Created file")
        assert step.step_number == 1
        assert step.action == "Created file"
        assert step.description == ""
        assert step.tool_used is None
        assert step.result_summary is None
        assert step.files_involved == []
        assert step.success is True
        assert isinstance(step.timestamp, datetime)

    def test_create_full(self):
        step = StepRecord(
            step_number=5,
            action="Ran test suite",
            description="Executed pytest with coverage",
            tool_used="Bash",
            result_summary="12 tests passed, 0 failed",
            files_involved=["tests/test_models.py"],
            success=True,
        )
        assert step.step_number == 5
        assert step.tool_used == "Bash"
        assert step.result_summary == "12 tests passed, 0 failed"
        assert step.files_involved == ["tests/test_models.py"]

    def test_json_round_trip(self):
        step = StepRecord(
            step_number=1,
            action="Edited module",
            tool_used="Edit",
            result_summary="Line replaced",
        )
        json_data = step.model_dump(mode="json")
        restored = StepRecord.model_validate(json_data)
        assert restored.step_number == step.step_number
        assert restored.action == step.action
        assert restored.tool_used == step.tool_used

    def test_step_number_must_be_positive(self):
        import pytest

        with pytest.raises(Exception):
            StepRecord(step_number=0, action="Invalid")

    def test_action_cannot_be_empty(self):
        import pytest

        with pytest.raises(Exception):
            StepRecord(step_number=1, action="")


# ---------------------------------------------------------------------------
# FileRecord Tests
# ---------------------------------------------------------------------------


class TestFileRecord:
    def test_create_minimal(self):
        record = FileRecord(path="src/models.py", action=FileAction.CREATED)
        assert record.path == "src/models.py"
        assert record.action == FileAction.CREATED
        assert record.description == ""
        assert record.action_history == []

    def test_add_action_deduplication(self):
        record = FileRecord(
            path="src/models.py",
            action=FileAction.CREATED,
            description="Initial creation",
        )
        record.add_action(FileAction.MODIFIED, "Added new class")
        assert record.action == FileAction.MODIFIED
        assert record.description == "Added new class"
        assert len(record.action_history) == 1
        assert record.action_history[0]["action"] == "created"
        assert record.action_history[0]["description"] == "Initial creation"

    def test_multiple_actions(self):
        record = FileRecord(path="config.json", action=FileAction.CREATED)
        record.add_action(FileAction.MODIFIED, "Updated settings")
        record.add_action(FileAction.MODIFIED, "Fixed typo")
        assert len(record.action_history) == 2
        assert record.action == FileAction.MODIFIED

    def test_json_round_trip(self):
        record = FileRecord(
            path="src/app.py",
            action=FileAction.MODIFIED,
            description="Refactored",
        )
        json_data = record.model_dump(mode="json")
        restored = FileRecord.model_validate(json_data)
        assert restored.path == record.path
        assert restored.action == record.action

    def test_file_action_enum_values(self):
        assert FileAction.CREATED.value == "created"
        assert FileAction.MODIFIED.value == "modified"
        assert FileAction.DELETED.value == "deleted"
        assert FileAction.RENAMED.value == "renamed"
        assert FileAction.READ.value == "read"


# ---------------------------------------------------------------------------
# BranchRecord Tests
# ---------------------------------------------------------------------------


class TestBranchRecord:
    def test_create_minimal(self):
        record = BranchRecord(
            branch_name="feature/CMH-002-models",
            action=BranchAction.CREATED,
            base_branch="main",
        )
        assert record.branch_name == "feature/CMH-002-models"
        assert record.action == BranchAction.CREATED
        assert record.base_branch == "main"

    def test_add_action(self):
        record = BranchRecord(
            branch_name="feature/CMH-002-models",
            action=BranchAction.CREATED,
            base_branch="main",
        )
        record.add_action(BranchAction.PUSHED)
        assert record.action == BranchAction.PUSHED
        assert len(record.action_history) == 1
        assert record.action_history[0]["action"] == "created"

    def test_branch_action_enum_values(self):
        assert BranchAction.CREATED.value == "created"
        assert BranchAction.CHECKED_OUT.value == "checked_out"
        assert BranchAction.MERGED.value == "merged"
        assert BranchAction.DELETED.value == "deleted"
        assert BranchAction.PUSHED.value == "pushed"
        assert BranchAction.PULLED.value == "pulled"

    def test_json_round_trip(self):
        record = BranchRecord(
            branch_name="fix/CMH-010-bug",
            action=BranchAction.CREATED,
            base_branch="main",
        )
        json_data = record.model_dump(mode="json")
        restored = BranchRecord.model_validate(json_data)
        assert restored.branch_name == record.branch_name
        assert restored.action == record.action


# ---------------------------------------------------------------------------
# DecisionRecord Tests
# ---------------------------------------------------------------------------


class TestDecisionRecord:
    def test_create_minimal(self):
        record = DecisionRecord(
            decision_number=1,
            decision="Use Pydantic v2 for data models",
        )
        assert record.decision_number == 1
        assert record.decision == "Use Pydantic v2 for data models"
        assert record.reasoning == ""
        assert record.alternatives == []
        assert record.context == ""

    def test_create_full(self):
        record = DecisionRecord(
            decision_number=2,
            decision="Use file-based storage instead of SQLite",
            reasoning="Portability and simplicity are more important than query power",
            alternatives=["SQLite", "TinyDB", "JSON in-memory"],
            context="Project brief specifies no external databases",
        )
        assert len(record.alternatives) == 3
        assert "SQLite" in record.alternatives

    def test_json_round_trip(self):
        record = DecisionRecord(
            decision_number=1,
            decision="Test decision",
            reasoning="Test reasoning",
            alternatives=["alt1", "alt2"],
        )
        json_data = record.model_dump(mode="json")
        restored = DecisionRecord.model_validate(json_data)
        assert restored.decision == record.decision
        assert restored.alternatives == record.alternatives


# ---------------------------------------------------------------------------
# TaskMemory Tests
# ---------------------------------------------------------------------------


class TestTaskMemory:
    def test_create_minimal(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        assert task.ticket_id == "CMH-002"
        assert task.title == "Data models"
        assert task.status == TaskStatus.ACTIVE
        assert task.steps == []
        assert task.files == []
        assert task.branches == []
        assert task.decisions == []
        assert task.completed_at is None

    def test_add_step(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        step = task.add_step("Created records.py", tool_used="Write")
        assert step.step_number == 1
        assert len(task.steps) == 1

        step2 = task.add_step("Added tests", tool_used="Write")
        assert step2.step_number == 2
        assert len(task.steps) == 2

    def test_record_file_deduplication(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.record_file("src/models.py", FileAction.CREATED, "Initial file")
        task.record_file("src/models.py", FileAction.MODIFIED, "Added class")
        assert len(task.files) == 1  # Deduplicated
        assert task.files[0].action == FileAction.MODIFIED
        assert len(task.files[0].action_history) == 1

    def test_record_file_different_paths(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.record_file("src/a.py", FileAction.CREATED)
        task.record_file("src/b.py", FileAction.CREATED)
        assert len(task.files) == 2

    def test_record_branch_deduplication(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.record_branch("feature/CMH-002", BranchAction.CREATED, "main")
        task.record_branch("feature/CMH-002", BranchAction.PUSHED)
        assert len(task.branches) == 1
        assert task.branches[0].action == BranchAction.PUSHED

    def test_add_decision(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        d1 = task.add_decision("Use Pydantic v2", reasoning="Better performance")
        d2 = task.add_decision("Use enums for actions", reasoning="Type safety")
        assert d1.decision_number == 1
        assert d2.decision_number == 2
        assert len(task.decisions) == 2

    def test_complete(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.complete("All models implemented and tested")
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.summary == "All models implemented and tested"

    def test_get_file_paths(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.record_file("src/a.py", FileAction.CREATED)
        task.record_file("src/b.py", FileAction.MODIFIED)
        assert task.get_file_paths() == ["src/a.py", "src/b.py"]

    def test_get_active_branch(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        assert task.get_active_branch() is None
        task.record_branch("feature/CMH-002", BranchAction.CREATED, "main")
        assert task.get_active_branch() == "feature/CMH-002"

    def test_step_count(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        assert task.step_count() == 0
        task.add_step("Step 1")
        task.add_step("Step 2")
        assert task.step_count() == 2

    def test_json_round_trip(self):
        task = TaskMemory(
            ticket_id="CMH-002",
            title="Data models",
            phase="phase-1",
        )
        task.add_step("Created file", tool_used="Write")
        task.record_file("src/models.py", FileAction.CREATED)
        task.record_branch("feature/CMH-002", BranchAction.CREATED, "main")
        task.add_decision("Use Pydantic", reasoning="Schema validation")

        json_data = task.to_json_dict()
        restored = TaskMemory.from_json_dict(json_data)

        assert restored.ticket_id == "CMH-002"
        assert restored.title == "Data models"
        assert restored.phase == "phase-1"
        assert len(restored.steps) == 1
        assert len(restored.files) == 1
        assert len(restored.branches) == 1
        assert len(restored.decisions) == 1

    def test_json_file_round_trip(self):
        """Verify serialization to and from a real JSON file."""
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.add_step("Wrote code", tool_used="Write")
        task.record_file("src/models.py", FileAction.CREATED, "Initial")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(task.to_json_dict(), f, indent=2)
            tmp_path = f.name

        with open(tmp_path, "r") as f:
            loaded_data = json.load(f)

        restored = TaskMemory.from_json_dict(loaded_data)
        assert restored.ticket_id == "CMH-002"
        assert len(restored.steps) == 1
        assert restored.steps[0].tool_used == "Write"

        Path(tmp_path).unlink()

    def test_completion_validator(self):
        """Completed status auto-sets completed_at if missing."""
        task = TaskMemory(
            ticket_id="CMH-002",
            title="Data models",
            status=TaskStatus.COMPLETED,
        )
        assert task.completed_at is not None

    def test_next_steps(self):
        task = TaskMemory(ticket_id="CMH-002", title="Data models")
        task.next_steps = ["Implement storage engine", "Write tests"]
        assert len(task.next_steps) == 2

    def test_metadata(self):
        task = TaskMemory(
            ticket_id="CMH-002",
            title="Data models",
            metadata={"pr_number": 29, "github_issue": 2},
        )
        assert task.metadata["pr_number"] == 29

    def test_task_status_enum_values(self):
        assert TaskStatus.ACTIVE.value == "active"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.ARCHIVED.value == "archived"
        assert TaskStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# RecoveryContext Tests
# ---------------------------------------------------------------------------


class TestRecoveryContext:
    def _make_task_with_data(self) -> TaskMemory:
        """Create a TaskMemory populated with real test data."""
        task = TaskMemory(
            ticket_id="CMH-002",
            title="Memory data schema and models",
            phase="phase-1",
        )
        task.add_step("Created models directory", tool_used="Bash")
        task.add_step("Wrote records.py", tool_used="Write", result_summary="File created")
        task.add_step("Wrote task.py", tool_used="Write", result_summary="File created")
        task.record_file("src/models/records.py", FileAction.CREATED)
        task.record_file("src/models/task.py", FileAction.CREATED)
        task.record_branch("feature/CMH-002", BranchAction.CREATED, "main")
        task.add_decision(
            "Use Pydantic v2 BaseModel",
            reasoning="Provides validation, serialization, and JSON schema generation",
            alternatives=["dataclasses", "attrs", "plain dicts"],
        )
        task.next_steps = [
            "Write RecoveryContext model",
            "Write MemoryWindow model",
            "Add comprehensive tests",
        ]
        return task

    def test_from_task_memory(self):
        task = self._make_task_with_data()
        ctx = RecoveryContext.from_task_memory(task)

        assert ctx.ticket_id == "CMH-002"
        assert ctx.title == "Memory data schema and models"
        assert ctx.phase == "phase-1"
        assert ctx.status == "active"
        assert ctx.total_steps_completed == 3
        assert len(ctx.files_modified) == 2
        assert ctx.active_branch == "feature/CMH-002"
        assert len(ctx.key_decisions) == 1
        assert len(ctx.next_steps) == 3
        assert ctx.last_step is not None
        assert ctx.last_step["action"] == "Wrote task.py"

    def test_recent_steps_order(self):
        task = self._make_task_with_data()
        ctx = RecoveryContext.from_task_memory(task, recent_step_count=2)
        # Most recent first
        assert len(ctx.recent_steps) == 2
        assert ctx.recent_steps[0]["action"] == "Wrote task.py"
        assert ctx.recent_steps[1]["action"] == "Wrote records.py"

    def test_json_round_trip(self):
        task = self._make_task_with_data()
        ctx = RecoveryContext.from_task_memory(task)
        json_data = ctx.to_json_dict()
        restored = RecoveryContext.from_json_dict(json_data)

        assert restored.ticket_id == ctx.ticket_id
        assert restored.total_steps_completed == ctx.total_steps_completed
        assert restored.files_modified == ctx.files_modified

    def test_format_for_prompt(self):
        task = self._make_task_with_data()
        ctx = RecoveryContext.from_task_memory(task)
        prompt = ctx.format_for_prompt()

        assert "CMH-002" in prompt
        assert "Memory data schema and models" in prompt
        assert "phase-1" in prompt
        assert "feature/CMH-002" in prompt
        assert "Pydantic v2" in prompt
        assert "Write RecoveryContext model" in prompt

    def test_empty_task_recovery(self):
        task = TaskMemory(ticket_id="CMH-010", title="Empty task")
        ctx = RecoveryContext.from_task_memory(task)

        assert ctx.ticket_id == "CMH-010"
        assert ctx.total_steps_completed == 0
        assert ctx.last_step is None
        assert ctx.files_modified == []
        assert ctx.active_branch is None


# ---------------------------------------------------------------------------
# MemoryWindow Tests
# ---------------------------------------------------------------------------


class TestMemoryWindow:
    def test_create_default(self):
        window = MemoryWindow()
        assert window.window_size == 3
        assert window.current_task is None
        assert window.completed_tasks == []
        assert window.archived_task_ids == []

    def test_start_task(self):
        window = MemoryWindow()
        task = window.start_task("CMH-002", "Data models", "phase-1")
        assert task.ticket_id == "CMH-002"
        assert window.current_task is task
        assert window.has_active_task() is True

    def test_cannot_start_task_while_active(self):
        import pytest

        window = MemoryWindow()
        window.start_task("CMH-002", "Data models")
        with pytest.raises(ValueError, match="still active"):
            window.start_task("CMH-003", "Storage engine")

    def test_complete_current_task(self):
        window = MemoryWindow()
        window.start_task("CMH-002", "Data models")
        completed = window.complete_current_task("All done")
        assert completed.status == TaskStatus.COMPLETED
        assert completed.summary == "All done"
        assert window.current_task is None
        assert len(window.completed_tasks) == 1
        assert window.has_active_task() is False

    def test_complete_no_task_raises(self):
        import pytest

        window = MemoryWindow()
        with pytest.raises(ValueError, match="No current task"):
            window.complete_current_task()

    def test_fail_current_task(self):
        window = MemoryWindow()
        window.start_task("CMH-002", "Data models")
        failed = window.fail_current_task("Build compilation error")
        assert failed.status == TaskStatus.FAILED
        assert "FAILED:" in failed.summary
        assert window.current_task is None
        assert len(window.completed_tasks) == 1

    def test_sliding_window_enforcement(self):
        window = MemoryWindow(window_size=2)

        # Complete 3 tasks (exceeds window_size of 2)
        window.start_task("T-001", "Task 1")
        window.complete_current_task("Done 1")

        window.start_task("T-002", "Task 2")
        window.complete_current_task("Done 2")

        window.start_task("T-003", "Task 3")
        window.complete_current_task("Done 3")

        assert len(window.completed_tasks) == 2
        assert window.completed_tasks[0].ticket_id == "T-002"
        assert window.completed_tasks[1].ticket_id == "T-003"
        assert "T-001" in window.archived_task_ids

    def test_get_task_current(self):
        window = MemoryWindow()
        window.start_task("CMH-002", "Data models")
        found = window.get_task("CMH-002")
        assert found is not None
        assert found.ticket_id == "CMH-002"

    def test_get_task_completed(self):
        window = MemoryWindow()
        window.start_task("CMH-001", "Init")
        window.complete_current_task("Done")
        found = window.get_task("CMH-001")
        assert found is not None
        assert found.ticket_id == "CMH-001"

    def test_get_task_not_found(self):
        window = MemoryWindow()
        found = window.get_task("CMH-999")
        assert found is None

    def test_get_all_task_ids(self):
        window = MemoryWindow()
        window.start_task("T-001", "Task 1")
        window.complete_current_task()
        window.start_task("T-002", "Task 2")
        ids = window.get_all_task_ids()
        assert "T-001" in ids
        assert "T-002" in ids

    def test_is_task_archived(self):
        window = MemoryWindow(window_size=1)
        window.start_task("T-001", "Task 1")
        window.complete_current_task()
        window.start_task("T-002", "Task 2")
        window.complete_current_task()
        assert window.is_task_archived("T-001") is True
        assert window.is_task_archived("T-002") is False

    def test_total_tasks_in_window(self):
        window = MemoryWindow()
        assert window.total_tasks_in_window() == 0
        window.start_task("T-001", "Task 1")
        assert window.total_tasks_in_window() == 1
        window.complete_current_task()
        assert window.total_tasks_in_window() == 1
        window.start_task("T-002", "Task 2")
        assert window.total_tasks_in_window() == 2

    def test_json_round_trip(self):
        window = MemoryWindow(window_size=3)
        window.start_task("CMH-001", "Init", "phase-1")
        window.current_task.add_step("Setup project", tool_used="Bash")
        window.complete_current_task("Project initialized")
        window.start_task("CMH-002", "Models", "phase-1")

        json_data = window.to_json_dict()
        restored = MemoryWindow.from_json_dict(json_data)

        assert restored.window_size == 3
        assert len(restored.completed_tasks) == 1
        assert restored.completed_tasks[0].ticket_id == "CMH-001"
        assert restored.current_task is not None
        assert restored.current_task.ticket_id == "CMH-002"

    def test_json_file_round_trip(self):
        """Full round trip through a real JSON file on disk."""
        window = MemoryWindow(window_size=2)
        window.start_task("T-001", "Task 1")
        window.current_task.add_step("Did work", tool_used="Edit")
        window.complete_current_task("Completed")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(window.to_json_dict(), f, indent=2)
            tmp_path = f.name

        with open(tmp_path, "r") as f:
            loaded_data = json.load(f)

        restored = MemoryWindow.from_json_dict(loaded_data)
        assert len(restored.completed_tasks) == 1
        assert restored.completed_tasks[0].ticket_id == "T-001"
        assert restored.completed_tasks[0].steps[0].tool_used == "Edit"

        Path(tmp_path).unlink()

    def test_window_size_enforcement_on_load(self):
        """Window validator trims excess completed tasks on deserialization."""
        data = {
            "window_size": 1,
            "completed_tasks": [
                {"ticket_id": "T-001", "title": "Old task", "status": "completed"},
                {"ticket_id": "T-002", "title": "New task", "status": "completed"},
            ],
        }
        window = MemoryWindow.from_json_dict(data)
        assert len(window.completed_tasks) == 1
        assert window.completed_tasks[0].ticket_id == "T-002"
        assert "T-001" in window.archived_task_ids

    def test_updated_at_changes(self):
        window = MemoryWindow()
        initial_time = window.updated_at
        window.start_task("T-001", "Task")
        assert window.updated_at >= initial_time
