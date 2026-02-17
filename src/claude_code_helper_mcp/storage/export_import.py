"""Memory export and import functionality for portable memory sharing.

Provides ExportManager for exporting task memory to portable JSON with format
versioning, and ImportManager for importing with validation, merge/replace
options, and compatibility checks.

Typical usage::

    # Export
    exporter = ExportManager(storage_path="/path/to/project/.claude-memory")
    exporter.export_all(output_path="/path/to/export.json")
    exporter.export_ticket("CMH-003", output_path="/path/to/cmh003.json")

    # Import
    importer = ImportManager(storage_path="/path/to/project/.claude-memory")
    result = importer.import_from_file(
        file_path="/path/to/import.json",
        mode="merge",  # or "replace"
        validate_compatibility=True
    )
    if result["status"] == "success":
        print(f"Imported {result['imported_count']} tasks")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from claude_code_helper_mcp import __version__
from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager

logger = logging.getLogger(__name__)

# Export format version for compatibility checks
EXPORT_FORMAT_VERSION = "1.0.0"


class ExportManager:
    """Manages memory export to portable JSON format.

    Supports exporting all tasks or specific tickets to portable JSON with:
    - Format version for forward compatibility
    - Metadata (export timestamp, exporter version, source path)
    - All task data including steps, files, branches, decisions

    Parameters
    ----------
    storage_path:
        Path to the .claude-memory storage directory. When None, uses default.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._store = MemoryStore(storage_path)
        self._manager = WindowManager(storage_path=storage_path)

    def export_all(self, output_path: str) -> dict:
        """Export all tasks (current, completed, and archived) to JSON.

        Parameters
        ----------
        output_path:
            Path to write the export file to.

        Returns
        -------
        dict
            Result dictionary with keys:
            - status: "success" or "error"
            - file_path: Path to the exported file
            - exported_count: Number of tasks exported
            - format_version: Export format version
            - timestamp: Export timestamp
            - error: Error message (if status is "error")
        """
        try:
            tasks_to_export: list[TaskMemory] = []

            # Collect current task
            current = self._manager.get_current_task()
            if current is not None:
                tasks_to_export.append(current)

            # Collect completed tasks from window
            for task in self._manager.window.completed_tasks:
                tasks_to_export.append(task)

            # Collect archived tasks
            for archived_id in self._manager.window.archived_task_ids:
                archived_task = self._store.load_task(archived_id)
                if archived_task is not None:
                    tasks_to_export.append(archived_task)

            # Build export data
            export_data = {
                "format_version": EXPORT_FORMAT_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "exporter_version": __version__,
                "source_storage_path": str(self._store.storage_root),
                "tasks": [task.to_json_dict() for task in tasks_to_export],
                "summary": {
                    "total_tasks": len(tasks_to_export),
                    "current_task_count": 1 if current else 0,
                    "completed_tasks_count": len(self._manager.window.completed_tasks),
                    "archived_tasks_count": len(self._manager.window.archived_task_ids),
                },
            }

            # Write to file atomically
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                f.write("\n")

            logger.info(
                "Exported %d tasks to %s",
                len(tasks_to_export),
                output_file,
            )

            return {
                "status": "success",
                "file_path": str(output_file),
                "exported_count": len(tasks_to_export),
                "format_version": EXPORT_FORMAT_VERSION,
                "timestamp": export_data["exported_at"],
            }

        except Exception as exc:
            logger.exception("Export failed")
            return {
                "status": "error",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def export_ticket(self, ticket_id: str, output_path: str) -> dict:
        """Export a single task by ticket ID to JSON.

        Parameters
        ----------
        ticket_id:
            The ticket ID to export (e.g., "CMH-003").
        output_path:
            Path to write the export file to.

        Returns
        -------
        dict
            Result dictionary with keys:
            - status: "success", "not_found", or "error"
            - file_path: Path to the exported file
            - format_version: Export format version
            - timestamp: Export timestamp
            - error: Error message (if status is not "success")
        """
        try:
            # Try to load the task (window first, then disk)
            task = self._manager.get_task(ticket_id)
            if task is None:
                return {
                    "status": "not_found",
                    "error": f"Task '{ticket_id}' not found",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Build export data
            export_data = {
                "format_version": EXPORT_FORMAT_VERSION,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "exporter_version": __version__,
                "source_storage_path": str(self._store.storage_root),
                "tasks": [task.to_json_dict()],
                "summary": {
                    "total_tasks": 1,
                    "ticket_id": ticket_id,
                },
            }

            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                f.write("\n")

            logger.info("Exported task %s to %s", ticket_id, output_file)

            return {
                "status": "success",
                "file_path": str(output_file),
                "exported_count": 1,
                "format_version": EXPORT_FORMAT_VERSION,
                "timestamp": export_data["exported_at"],
            }

        except Exception as exc:
            logger.exception("Export failed for task %s", ticket_id)
            return {
                "status": "error",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


class ImportManager:
    """Manages memory import from portable JSON format.

    Supports importing tasks with:
    - Format version compatibility checking
    - Schema validation
    - Merge (add alongside existing) or replace (overwrite) modes
    - Duplicate detection
    - Conflict resolution

    Parameters
    ----------
    storage_path:
        Path to the .claude-memory storage directory. When None, uses default.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._store = MemoryStore(storage_path)
        self._manager = WindowManager(storage_path=storage_path)

    def import_from_file(
        self,
        file_path: str,
        mode: str = "merge",
        validate_compatibility: bool = True,
        allow_version_mismatch: bool = False,
    ) -> dict:
        """Import tasks from an exported JSON file.

        Parameters
        ----------
        file_path:
            Path to the import file.
        mode:
            Import mode: "merge" (add alongside existing) or "replace" (overwrite
            matching tasks). Default is "merge".
        validate_compatibility:
            If True, check format version compatibility. Default is True.
        allow_version_mismatch:
            If True, allow importing from different format versions. Only applies
            if validate_compatibility is True. Default is False.

        Returns
        -------
        dict
            Result dictionary with keys:
            - status: "success", "validation_error", "compatibility_error", or "error"
            - imported_count: Number of tasks successfully imported
            - skipped_count: Number of tasks skipped (duplicates or conflicts)
            - format_version: Format version of imported file
            - timestamp: Import timestamp
            - validation_errors: List of validation errors (if any)
            - error: Error message (if status is not "success")
        """
        try:
            # Read import file
            import_file = Path(file_path)
            if not import_file.exists():
                return {
                    "status": "error",
                    "error": f"Import file not found: {file_path}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            with open(import_file, "r", encoding="utf-8") as f:
                import_data = json.load(f)

        except json.JSONDecodeError as exc:
            return {
                "status": "error",
                "error": f"Invalid JSON in import file: {exc}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.exception("Failed to read import file")
            return {
                "status": "error",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Validate format version
        import_format_version = import_data.get("format_version")
        if validate_compatibility:
            if import_format_version != EXPORT_FORMAT_VERSION:
                if not allow_version_mismatch:
                    return {
                        "status": "compatibility_error",
                        "error": (
                            f"Format version mismatch: import has {import_format_version}, "
                            f"expected {EXPORT_FORMAT_VERSION}"
                        ),
                        "import_format_version": import_format_version,
                        "expected_format_version": EXPORT_FORMAT_VERSION,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                logger.warning(
                    "Importing from different format version: %s (expected %s)",
                    import_format_version,
                    EXPORT_FORMAT_VERSION,
                )

        # Validate tasks structure
        tasks_data = import_data.get("tasks", [])
        if not isinstance(tasks_data, list):
            return {
                "status": "validation_error",
                "error": "Invalid import structure: 'tasks' must be a list",
                "validation_errors": ["'tasks' field must be an array"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Validate and import each task
        imported_count = 0
        skipped_count = 0
        validation_errors: list[str] = []

        for task_data in tasks_data:
            ticket_id = task_data.get("ticket_id")
            if not ticket_id:
                validation_errors.append("Task missing ticket_id field")
                skipped_count += 1
                continue

            # Validate task schema
            try:
                task = TaskMemory.from_json_dict(task_data)
            except Exception as exc:
                validation_errors.append(
                    f"Task {ticket_id}: Invalid schema - {str(exc)}"
                )
                skipped_count += 1
                continue

            # Check for duplicates
            existing = self._manager.get_task(ticket_id)
            if existing is not None:
                if mode == "merge":
                    logger.info(
                        "Skipping duplicate task %s (mode=merge)",
                        ticket_id,
                    )
                    skipped_count += 1
                    continue
                elif mode == "replace":
                    logger.info(
                        "Replacing existing task %s (mode=replace)",
                        ticket_id,
                    )

            # Save the task
            try:
                self._store.save_task(task)
                imported_count += 1
                logger.info("Imported task %s", ticket_id)
            except Exception as exc:
                validation_errors.append(f"Task {ticket_id}: Failed to save - {str(exc)}")
                skipped_count += 1

        # Determine overall status
        if imported_count == 0 and skipped_count > 0:
            status = "validation_error" if validation_errors else "success"
        else:
            status = "success"

        return {
            "status": status,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "format_version": import_format_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_errors": validation_errors if validation_errors else None,
        }

    def validate_import_file(self, file_path: str) -> dict:
        """Validate an import file without importing.

        Parameters
        ----------
        file_path:
            Path to the import file.

        Returns
        -------
        dict
            Validation result with keys:
            - status: "valid" or "invalid"
            - format_version: Format version in file
            - task_count: Number of tasks in file
            - validation_errors: List of validation errors (if any)
            - error: Error message (if status is "invalid")
        """
        try:
            import_file = Path(file_path)
            if not import_file.exists():
                return {
                    "status": "invalid",
                    "error": f"File not found: {file_path}",
                }

            with open(import_file, "r", encoding="utf-8") as f:
                import_data = json.load(f)

        except json.JSONDecodeError as exc:
            return {
                "status": "invalid",
                "error": f"Invalid JSON: {exc}",
            }
        except Exception as exc:
            return {
                "status": "invalid",
                "error": str(exc),
            }

        # Validate structure
        validation_errors: list[str] = []

        import_format_version = import_data.get("format_version")
        if not import_format_version:
            validation_errors.append("Missing format_version field")

        if not isinstance(import_data.get("tasks"), list):
            validation_errors.append("'tasks' field must be an array")

        tasks_data = import_data.get("tasks", [])
        for i, task_data in enumerate(tasks_data):
            if not task_data.get("ticket_id"):
                validation_errors.append(f"Task {i}: Missing ticket_id field")
                continue

            try:
                TaskMemory.from_json_dict(task_data)
            except Exception as exc:
                validation_errors.append(
                    f"Task {task_data.get('ticket_id')}: Invalid schema - {str(exc)}"
                )

        status = "valid" if not validation_errors else "invalid"

        return {
            "status": status,
            "format_version": import_format_version,
            "task_count": len(tasks_data),
            "validation_errors": validation_errors if validation_errors else None,
        }
