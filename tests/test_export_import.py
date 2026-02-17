"""Tests for memory export and import functionality.

Tests cover:
- ExportManager: exporting all tasks, single tasks, format version, metadata
- ImportManager: importing with validation, merge/replace modes, compatibility checks
- CLI commands: memory export, memory import with various options
- Format validation and error handling
- Duplicate detection and conflict resolution
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.export_import import (
    EXPORT_FORMAT_VERSION,
    ExportManager,
    ImportManager,
)
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


class TestExportManager:
    """Tests for ExportManager."""

    def test_export_all_empty_storage(self, tmp_path):
        """Test exporting from empty storage creates valid export file."""
        storage_path = str(tmp_path / ".claude-memory")
        exporter = ExportManager(storage_path)

        output_file = str(tmp_path / "export.json")
        result = exporter.export_all(output_file)

        assert result["status"] == "success"
        assert result["exported_count"] == 0
        assert result["format_version"] == EXPORT_FORMAT_VERSION
        assert Path(output_file).exists()

        # Verify export file structure
        with open(output_file) as f:
            data = json.load(f)
        assert data["format_version"] == EXPORT_FORMAT_VERSION
        assert "exported_at" in data
        assert "exporter_version" in data
        assert data["tasks"] == []

    def test_export_single_task(self, tmp_path):
        """Test exporting a single task by ticket ID."""
        storage_path = str(tmp_path / ".claude-memory")
        manager = WindowManager(storage_path)

        # Create and save a task
        task = manager.start_new_task("CMH-001", "Test task", phase="phase-1")
        task.add_step("Step 1", "Testing")
        manager.save_current_task()

        # Export the single task
        exporter = ExportManager(storage_path)
        output_file = str(tmp_path / "single.json")
        result = exporter.export_ticket("CMH-001", output_file)

        assert result["status"] == "success"
        assert result["exported_count"] == 1
        assert Path(output_file).exists()

        with open(output_file) as f:
            data = json.load(f)
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["ticket_id"] == "CMH-001"

    def test_export_nonexistent_ticket(self, tmp_path):
        """Test exporting nonexistent ticket returns not_found."""
        storage_path = str(tmp_path / ".claude-memory")
        exporter = ExportManager(storage_path)

        output_file = str(tmp_path / "nonexistent.json")
        result = exporter.export_ticket("CMH-999", output_file)

        assert result["status"] == "not_found"
        assert "not found" in result["error"].lower()

    def test_export_all_with_multiple_tasks(self, tmp_path):
        """Test exporting all tasks including current, completed, and archived."""
        storage_path = str(tmp_path / ".claude-memory")
        manager = WindowManager(storage_path, window_size=2)
        store = MemoryStore(storage_path)

        # Create first task and complete it
        task1 = manager.start_new_task("CMH-001", "First task", phase="phase-1")
        task1.add_step("Worked on it")
        manager.complete_current_task("Done")

        # Create second task and complete it
        task2 = manager.start_new_task("CMH-002", "Second task", phase="phase-1")
        task2.add_step("Worked on it")
        manager.complete_current_task("Done")

        # Create third task and complete it (this should archive task 1)
        task3 = manager.start_new_task("CMH-003", "Third task", phase="phase-1")
        task3.add_step("Worked on it")
        manager.complete_current_task("Done")

        # Now we should have: task3 current, task2+task3 completed, task1 archived
        # Create current task for export
        task4 = manager.start_new_task("CMH-004", "Fourth task", phase="phase-1")

        exporter = ExportManager(storage_path)
        output_file = str(tmp_path / "all.json")
        result = exporter.export_all(output_file)

        assert result["status"] == "success"
        assert result["exported_count"] >= 2  # At least current and one completed

        with open(output_file) as f:
            data = json.load(f)
        assert len(data["tasks"]) == result["exported_count"]

    def test_export_format_version_included(self, tmp_path):
        """Test that exported format includes version for forward compatibility."""
        storage_path = str(tmp_path / ".claude-memory")
        manager = WindowManager(storage_path)
        task = manager.start_new_task("CMH-001", "Test", phase="phase-1")
        manager.save_current_task()

        exporter = ExportManager(storage_path)
        output_file = str(tmp_path / "versioned.json")
        exporter.export_all(output_file)

        with open(output_file) as f:
            data = json.load(f)

        assert data["format_version"] == EXPORT_FORMAT_VERSION
        assert data["exported_at"]
        assert data["exporter_version"]
        assert data["source_storage_path"]

    def test_export_metadata_included(self, tmp_path):
        """Test that export includes metadata."""
        storage_path = str(tmp_path / ".claude-memory")
        exporter = ExportManager(storage_path)
        output_file = str(tmp_path / "with_metadata.json")
        result = exporter.export_all(output_file)

        with open(output_file) as f:
            data = json.load(f)

        assert "summary" in data
        assert data["summary"]["total_tasks"] == 0


class TestImportManager:
    """Tests for ImportManager."""

    def test_import_valid_file_merge_mode(self, tmp_path):
        """Test importing a valid export file in merge mode."""
        # Create export
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task_src = manager_src.start_new_task("CMH-001", "Test task", phase="phase-1")
        task_src.add_step("Step 1")
        manager_src.save_current_task()

        exporter = ExportManager(storage_path_src)
        export_file = str(tmp_path / "export.json")
        exporter.export_all(export_file)

        # Import into different storage
        storage_path_dst = str(tmp_path / ".claude-memory-dst")
        importer = ImportManager(storage_path_dst)
        result = importer.import_from_file(export_file, mode="merge")

        assert result["status"] == "success"
        assert result["imported_count"] == 1

        # Verify task was imported
        store_dst = MemoryStore(storage_path_dst)
        imported_task = store_dst.load_task("CMH-001")
        assert imported_task is not None
        assert imported_task.ticket_id == "CMH-001"
        assert imported_task.title == "Test task"

    def test_import_merge_mode_skips_duplicates(self, tmp_path):
        """Test that merge mode skips duplicate tasks."""
        storage_path = str(tmp_path / ".claude-memory")
        manager = WindowManager(storage_path)

        # Create initial task
        task = manager.start_new_task("CMH-001", "Original", phase="phase-1")
        manager.save_current_task()

        # Export it
        exporter = ExportManager(storage_path)
        export_file = str(tmp_path / "export.json")
        exporter.export_all(export_file)

        # Try to import it back in merge mode
        importer = ImportManager(storage_path)
        result = importer.import_from_file(export_file, mode="merge")

        assert result["status"] == "success"
        assert result["imported_count"] == 0  # Should skip existing task
        assert result["skipped_count"] == 1

    def test_import_replace_mode_overwrites(self, tmp_path):
        """Test that replace mode overwrites existing tasks."""
        storage_path = str(tmp_path / ".claude-memory")
        manager = WindowManager(storage_path)

        # Create initial task with one step
        task = manager.start_new_task("CMH-001", "Original", phase="phase-1")
        task.add_step("Original step")
        manager.save_current_task()

        # Export it (before modification)
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task_src = manager_src.start_new_task("CMH-001", "Updated", phase="phase-2")
        task_src.add_step("Updated step 1")
        task_src.add_step("Updated step 2")
        manager_src.save_current_task()

        exporter = ExportManager(storage_path_src)
        export_file = str(tmp_path / "export.json")
        exporter.export_all(export_file)

        # Import in replace mode
        importer = ImportManager(storage_path)
        result = importer.import_from_file(export_file, mode="replace")

        assert result["status"] == "success"
        assert result["imported_count"] == 1

        # Verify task was replaced
        store = MemoryStore(storage_path)
        updated_task = store.load_task("CMH-001")
        assert updated_task.title == "Updated"
        assert updated_task.phase == "phase-2"
        assert len(updated_task.steps) == 2

    def test_import_invalid_json_file(self, tmp_path):
        """Test importing invalid JSON file returns error."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{invalid json}")

        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.import_from_file(str(invalid_file))

        assert result["status"] == "error"
        assert "Invalid JSON" in result["error"]

    def test_import_nonexistent_file(self, tmp_path):
        """Test importing nonexistent file returns error."""
        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.import_from_file(str(tmp_path / "nonexistent.json"))

        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_import_missing_format_version(self, tmp_path):
        """Test that missing format version is handled gracefully."""
        bad_export = tmp_path / "no_version.json"
        bad_export.write_text(json.dumps({"tasks": []}))

        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.import_from_file(str(bad_export))

        # Should still work but flag as compatibility issue
        assert result["status"] in ("success", "compatibility_error")

    def test_import_validates_task_schema(self, tmp_path):
        """Test that import validates task schema."""
        bad_export = tmp_path / "bad_schema.json"
        bad_export.write_text(json.dumps({
            "format_version": EXPORT_FORMAT_VERSION,
            "tasks": [
                {
                    # Missing required ticket_id field
                    "title": "No ticket ID",
                    "status": "active",
                }
            ]
        }))

        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.import_from_file(str(bad_export))

        assert result["status"] == "validation_error"
        assert len(result.get("validation_errors", [])) > 0

    def test_validate_import_file_valid(self, tmp_path):
        """Test validating a valid import file."""
        # Create valid export
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task_src = manager_src.start_new_task("CMH-001", "Test", phase="phase-1")
        manager_src.save_current_task()

        exporter = ExportManager(storage_path_src)
        export_file = str(tmp_path / "export.json")
        exporter.export_all(export_file)

        # Validate it
        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.validate_import_file(export_file)

        assert result["status"] == "valid"
        assert result["format_version"] == EXPORT_FORMAT_VERSION
        assert result["task_count"] == 1

    def test_validate_import_file_invalid(self, tmp_path):
        """Test validating an invalid import file."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({
            "format_version": EXPORT_FORMAT_VERSION,
            "tasks": [{"no_ticket_id": "problem"}]
        }))

        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.validate_import_file(str(bad_file))

        assert result["status"] == "invalid"
        assert len(result.get("validation_errors", [])) > 0

    def test_import_version_mismatch_blocked(self, tmp_path):
        """Test that version mismatch is blocked by default."""
        bad_export = tmp_path / "wrong_version.json"
        bad_export.write_text(json.dumps({
            "format_version": "0.9.0",  # Wrong version
            "tasks": []
        }))

        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.import_from_file(
            str(bad_export),
            validate_compatibility=True,
            allow_version_mismatch=False,
        )

        assert result["status"] == "compatibility_error"

    def test_import_version_mismatch_allowed(self, tmp_path):
        """Test that version mismatch can be allowed with flag."""
        # Create export
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task_src = manager_src.start_new_task("CMH-001", "Test", phase="phase-1")
        manager_src.save_current_task()

        exporter = ExportManager(storage_path_src)
        export_file = str(tmp_path / "export.json")
        exporter.export_all(export_file)

        # Manually change version
        with open(export_file) as f:
            data = json.load(f)
        data["format_version"] = "0.9.0"
        with open(export_file, "w") as f:
            json.dump(data, f)

        # Import with version mismatch allowed
        storage_path = str(tmp_path / ".claude-memory")
        importer = ImportManager(storage_path)
        result = importer.import_from_file(
            export_file,
            validate_compatibility=True,
            allow_version_mismatch=True,
        )

        assert result["status"] == "success"
        assert result["imported_count"] == 1

    def test_import_task_with_all_fields(self, tmp_path):
        """Test importing task with all fields (steps, files, branches, decisions)."""
        from claude_code_helper_mcp.models.records import FileAction, BranchAction

        # Create rich task
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task_src = manager_src.start_new_task("CMH-001", "Rich task", phase="phase-1")
        task_src.add_step("Step 1", "Did something")
        task_src.record_file("src/main.py", FileAction.CREATED)
        task_src.record_branch("feature/test", BranchAction.CREATED, "main")
        task_src.add_decision("Use Click for CLI", "Simple and Pythonic")
        manager_src.save_current_task()

        # Export and import
        exporter = ExportManager(storage_path_src)
        export_file = str(tmp_path / "export.json")
        exporter.export_all(export_file)

        storage_path_dst = str(tmp_path / ".claude-memory-dst")
        importer = ImportManager(storage_path_dst)
        result = importer.import_from_file(export_file)

        assert result["status"] == "success"

        # Verify all fields
        store_dst = MemoryStore(storage_path_dst)
        imported = store_dst.load_task("CMH-001")
        assert len(imported.steps) == 1
        assert len(imported.files) == 1
        assert len(imported.branches) == 1
        assert len(imported.decisions) == 1

    def test_import_multiple_tasks(self, tmp_path):
        """Test importing multiple tasks from single export."""
        # Create multiple tasks
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)

        for i in range(3):
            task = manager_src.start_new_task(f"CMH-{i:03d}", f"Task {i}", phase="phase-1")
            task.add_step(f"Step {i}")
            manager_src.complete_current_task()

        # Export all
        exporter = ExportManager(storage_path_src)
        export_file = str(tmp_path / "export.json")
        result = exporter.export_all(export_file)
        assert result["exported_count"] >= 3

        # Import all
        storage_path_dst = str(tmp_path / ".claude-memory-dst")
        importer = ImportManager(storage_path_dst)
        result = importer.import_from_file(export_file)

        assert result["status"] == "success"
        assert result["imported_count"] >= 3

        # Verify all were imported
        store_dst = MemoryStore(storage_path_dst)
        for i in range(3):
            task = store_dst.load_task(f"CMH-{i:03d}")
            assert task is not None


class TestExportImportCLI:
    """Tests for CLI export and import commands."""

    def test_export_command_basic(self, tmp_path, runner_cli):
        """Test basic export command."""
        storage_path = str(tmp_path / ".claude-memory")
        manager = WindowManager(storage_path)
        task = manager.start_new_task("CMH-001", "Test", phase="phase-1")
        manager.save_current_task()

        output_file = str(tmp_path / "export.json")
        result = runner_cli(
            ["--storage-path", storage_path, "export", output_file]
        )

        assert result.exit_code == 0
        assert "Export successful" in result.output
        assert Path(output_file).exists()

    def test_import_command_basic(self, tmp_path, runner_cli):
        """Test basic import command."""
        # Create and export
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task = manager_src.start_new_task("CMH-001", "Test", phase="phase-1")
        manager_src.save_current_task()

        export_file = str(tmp_path / "export.json")
        result = runner_cli(
            ["--storage-path", storage_path_src, "export", export_file]
        )
        assert result.exit_code == 0

        # Import
        storage_path_dst = str(tmp_path / ".claude-memory-dst")
        result = runner_cli(
            ["--storage-path", storage_path_dst, "import", export_file]
        )

        assert result.exit_code == 0
        assert "Import successful" in result.output

    def test_import_validate_only(self, tmp_path, runner_cli):
        """Test import --validate option."""
        # Create and export
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task = manager_src.start_new_task("CMH-001", "Test", phase="phase-1")
        manager_src.save_current_task()

        export_file = str(tmp_path / "export.json")
        runner_cli(
            ["--storage-path", storage_path_src, "export", export_file]
        )

        # Validate only
        result = runner_cli(
            ["import", export_file, "--validate"]
        )

        assert result.exit_code == 0
        assert "Validation passed" in result.output

    def test_import_json_output(self, tmp_path, runner_cli):
        """Test import --json-output option."""
        # Create and export
        storage_path_src = str(tmp_path / ".claude-memory-src")
        manager_src = WindowManager(storage_path_src)
        task = manager_src.start_new_task("CMH-001", "Test", phase="phase-1")
        manager_src.save_current_task()

        export_file = str(tmp_path / "export.json")
        runner_cli(
            ["--storage-path", storage_path_src, "export", export_file]
        )

        # Import with JSON output
        storage_path_dst = str(tmp_path / ".claude-memory-dst")
        result = runner_cli(
            ["--storage-path", storage_path_dst, "import", export_file, "--json-output"]
        )

        assert result.exit_code == 0
        # Should output JSON
        data = json.loads(result.output)
        assert "status" in data


@pytest.fixture
def runner_cli():
    """Provide Click CLI test runner."""
    from click.testing import CliRunner
    from claude_code_helper_mcp.cli.main import cli

    def run(args):
        runner = CliRunner()
        return runner.invoke(cli, args)

    return run
