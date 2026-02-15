"""Comprehensive tests for the ConfusionDetector (CMH-021).

Tests cover:
- Basic construction and project root validation
- Non-existent file reference detection (created, modified, read, deleted)
- Wrong function/class name detection against file content
- State contradiction detection (creation claims vs filesystem, fix+failure)
- Phantom step reference detection
- Severity classification (none, low, medium, high)
- TaskMemory integration via check_task()
- Report serialization (to_dict)
- Edge cases (empty inputs, no files, unreadable files, absolute paths)
- Levenshtein distance computation
- Name extraction from text (PascalCase, function_call())
- Defined name extraction from Python source
- Close match finding
- Path resolution (relative and absolute)
- Precision focus: no false positives on legitimate patterns

All tests use real filesystem I/O via tmp_path -- zero mocks.
"""

import os
from datetime import datetime, timezone

import pytest

from claude_code_helper_mcp.detection.confusion import (
    CONFUSION_SEVERITY_HIGH,
    CONFUSION_SEVERITY_LOW,
    CONFUSION_SEVERITY_MEDIUM,
    CONFUSION_SEVERITY_NONE,
    ConfusionDetector,
    ConfusionPattern,
    ConfusionReport,
    PATTERN_NONEXISTENT_FILE,
    PATTERN_PHANTOM_STEP,
    PATTERN_STATE_CONTRADICTION,
    PATTERN_WRONG_NAME,
    _is_likely_class_name,
    _levenshtein_distance,
)
from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.models.records import FileAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_detector(tmp_path) -> ConfusionDetector:
    """Create a ConfusionDetector rooted at tmp_path."""
    return ConfusionDetector(project_root=str(tmp_path))


def create_file(tmp_path, relative_path: str, content: str = "") -> str:
    """Create a file under tmp_path and return the relative path."""
    full = tmp_path / relative_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")
    return relative_path


def make_step(
    action: str,
    success: bool = True,
    files: list[str] | None = None,
    step_number: int = 1,
    result_summary: str = "",
    tool_used: str = "",
    description: str = "",
) -> dict:
    """Build a step record dict."""
    return {
        "action": action,
        "success": success,
        "files_involved": files or [],
        "step_number": step_number,
        "result_summary": result_summary,
        "tool_used": tool_used,
        "description": description,
    }


def make_file_record(path: str, action: str = "created") -> dict:
    """Build a file record dict."""
    return {"path": path, "action": action}


# ===================================================================
# Test class: Construction
# ===================================================================


class TestConstruction:
    """Test ConfusionDetector construction and validation."""

    def test_construction_with_valid_path(self, tmp_path):
        detector = ConfusionDetector(project_root=str(tmp_path))
        assert detector.project_root == str(tmp_path)

    def test_construction_normalises_path(self, tmp_path):
        # Trailing slash should be normalised.
        detector = ConfusionDetector(project_root=str(tmp_path) + "/")
        assert detector.project_root == str(tmp_path)

    def test_construction_empty_string_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            ConfusionDetector(project_root="")

    def test_construction_with_relative_path(self):
        # Relative paths are resolved to absolute.
        detector = ConfusionDetector(project_root="relative/path")
        assert os.path.isabs(detector.project_root)


# ===================================================================
# Test class: Empty inputs
# ===================================================================


class TestEmptyInputs:
    """Test behaviour with no records provided."""

    def test_empty_check(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check()
        assert not report.confusion_detected
        assert report.severity == CONFUSION_SEVERITY_NONE
        assert report.patterns == []
        assert report.total_files_checked == 0
        assert report.total_names_checked == 0
        assert report.total_steps_checked == 0

    def test_empty_file_records(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(file_records=[], step_records=[], branch_records=[])
        assert not report.confusion_detected
        assert report.severity == CONFUSION_SEVERITY_NONE

    def test_empty_task_memory(self, tmp_path):
        detector = make_detector(tmp_path)
        task = TaskMemory(ticket_id="CMH-TEST", title="Test task")
        report = detector.check_task(task)
        assert not report.confusion_detected
        assert report.severity == CONFUSION_SEVERITY_NONE


# ===================================================================
# Test class: Non-existent file detection
# ===================================================================


class TestNonExistentFileDetection:
    """Test detection of references to files that don't exist on disk."""

    def test_existing_file_no_pattern(self, tmp_path):
        create_file(tmp_path, "src/module.py", "# code")
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/module.py", "created")]
        )
        assert not report.confusion_detected
        assert report.total_files_checked == 1

    def test_created_file_missing(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/phantom.py", "created")]
        )
        assert report.confusion_detected
        assert len(report.patterns) == 1
        assert report.patterns[0].category == PATTERN_NONEXISTENT_FILE
        assert "phantom.py" in report.patterns[0].description
        assert report.patterns[0].severity_contribution == 0.4

    def test_modified_file_missing(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/gone.py", "modified")]
        )
        assert report.confusion_detected
        assert report.patterns[0].category == PATTERN_NONEXISTENT_FILE

    def test_deleted_file_not_flagged(self, tmp_path):
        """Deleted files are expected to be missing -- no false positive."""
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/removed.py", "deleted")]
        )
        assert not report.confusion_detected
        assert report.total_files_checked == 1

    def test_read_file_missing_low_severity(self, tmp_path):
        """Read files that are missing get flagged with lower severity."""
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/maybe.py", "read")]
        )
        assert report.confusion_detected
        assert report.patterns[0].severity_contribution == 0.15

    def test_multiple_missing_files(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[
                make_file_record("src/a.py", "created"),
                make_file_record("src/b.py", "modified"),
            ]
        )
        assert len(report.patterns) == 2
        assert report.total_files_checked == 2

    def test_step_files_involved_missing(self, tmp_path):
        """Files referenced in step files_involved should also be checked."""
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step("Edit module", files=["src/missing.py"])
            ]
        )
        assert report.confusion_detected
        assert report.patterns[0].category == PATTERN_NONEXISTENT_FILE
        assert "step files_involved" in report.patterns[0].description

    def test_step_file_exists_no_pattern(self, tmp_path):
        create_file(tmp_path, "src/exists.py", "# real")
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step("Edit module", files=["src/exists.py"])
            ]
        )
        # No nonexistent_file pattern.
        nonexist = [p for p in report.patterns if p.category == PATTERN_NONEXISTENT_FILE]
        assert len(nonexist) == 0

    def test_file_in_both_records_and_steps_not_double_counted(self, tmp_path):
        """If a file appears in both file_records and step files, check it once."""
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/one.py", "created")],
            step_records=[make_step("Edit", files=["src/one.py"])],
        )
        # Should only have 1 nonexistent pattern, not 2.
        nonexist = [p for p in report.patterns if p.category == PATTERN_NONEXISTENT_FILE]
        assert len(nonexist) == 1

    def test_renamed_file_missing(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/renamed.py", "renamed")]
        )
        assert report.confusion_detected
        assert report.patterns[0].category == PATTERN_NONEXISTENT_FILE


# ===================================================================
# Test class: Wrong name detection
# ===================================================================


class TestWrongNameDetection:
    """Test detection of references to names not defined in target files."""

    def test_correct_class_name_no_pattern(self, tmp_path):
        create_file(
            tmp_path,
            "src/detector.py",
            "class ConfusionDetector:\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Implemented ConfusionDetector class",
                    files=["src/detector.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) == 0

    def test_wrong_class_name_detected(self, tmp_path):
        create_file(
            tmp_path,
            "src/detector.py",
            "class ConfusionDetector:\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Implemented ConfusionChecker class",
                    files=["src/detector.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) >= 1
        assert "ConfusionChecker" in wrong[0].description

    def test_correct_function_name_no_pattern(self, tmp_path):
        create_file(
            tmp_path,
            "src/detector.py",
            "def check_alignment():\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Called check_alignment() function",
                    files=["src/detector.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) == 0

    def test_wrong_function_name_detected(self, tmp_path):
        create_file(
            tmp_path,
            "src/detector.py",
            "def check_alignment():\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Called check_aligment() function",
                    files=["src/detector.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) >= 1

    def test_no_files_involved_skipped(self, tmp_path):
        """Steps without files_involved are skipped for name checking."""
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step("Implemented ConfusionDetector class", files=[])
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) == 0
        assert report.total_names_checked == 0

    def test_non_python_file_skipped(self, tmp_path):
        """Non-Python files are not scanned for names."""
        create_file(tmp_path, "src/config.json", '{"key": "value"}')
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Updated ConfusionDetector config",
                    files=["src/config.json"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) == 0

    def test_close_match_suggestion(self, tmp_path):
        create_file(
            tmp_path,
            "src/store.py",
            "class MemoryStore:\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Updated MemoryStroe class",
                    files=["src/store.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) >= 1
        assert "MemoryStore" in wrong[0].description  # suggested correction

    def test_multiple_names_in_one_step(self, tmp_path):
        create_file(
            tmp_path,
            "src/module.py",
            "class TaskMemory:\n    pass\n\ndef record_step():\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Updated TaskMemory and called record_step()",
                    files=["src/module.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        # Both names exist in the file, so no wrong-name patterns.
        assert len(wrong) == 0

    def test_async_def_detected(self, tmp_path):
        """async def functions should be recognized."""
        create_file(
            tmp_path,
            "src/async_mod.py",
            "async def fetch_data():\n    pass\n",
        )
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Called fetch_data() async function",
                    files=["src/async_mod.py"],
                )
            ]
        )
        wrong = [p for p in report.patterns if p.category == PATTERN_WRONG_NAME]
        assert len(wrong) == 0


# ===================================================================
# Test class: State contradiction detection
# ===================================================================


class TestStateContradictions:
    """Test detection of contradictions between claims and filesystem state."""

    def test_created_file_exists_no_contradiction(self, tmp_path):
        create_file(tmp_path, "src/new.py", "# new file")
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Created src/new.py module",
                    success=True,
                    files=["src/new.py"],
                )
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 0

    def test_created_file_missing_contradiction(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Created src/phantom.py module",
                    success=True,
                    files=["src/phantom.py"],
                )
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 1
        assert "phantom.py" in contradictions[0].description
        assert contradictions[0].severity_contribution == 0.5

    def test_failed_step_no_contradiction(self, tmp_path):
        """Failed steps should not trigger creation contradictions."""
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Created src/phantom.py module",
                    success=False,
                    files=["src/phantom.py"],
                )
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 0

    def test_fix_then_failure_contradiction(self, tmp_path):
        create_file(tmp_path, "src/buggy.py", "# code with bug")
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Fixed the validation bug",
                    success=True,
                    files=["src/buggy.py"],
                    step_number=1,
                ),
                make_step(
                    "Ran tests",
                    success=False,
                    files=["src/buggy.py"],
                    step_number=2,
                    result_summary="Validation error in buggy.py",
                ),
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 1
        assert "step 1" in contradictions[0].description.lower() or "step 2" in contradictions[0].description.lower()

    def test_fix_then_success_no_contradiction(self, tmp_path):
        create_file(tmp_path, "src/fixed.py", "# fixed code")
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Fixed the bug",
                    success=True,
                    files=["src/fixed.py"],
                    step_number=1,
                ),
                make_step(
                    "Ran tests",
                    success=True,
                    files=["src/fixed.py"],
                    step_number=2,
                ),
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 0

    def test_wrote_keyword_triggers_creation_check(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Wrote the configuration file",
                    success=True,
                    files=["config.json"],
                )
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 1

    def test_non_overlapping_files_no_contradiction(self, tmp_path):
        create_file(tmp_path, "src/a.py", "# a")
        create_file(tmp_path, "src/b.py", "# b")
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Fixed bug in module a",
                    success=True,
                    files=["src/a.py"],
                    step_number=1,
                ),
                make_step(
                    "Ran tests for module b",
                    success=False,
                    files=["src/b.py"],
                    step_number=2,
                ),
            ]
        )
        contradictions = [
            p for p in report.patterns if p.category == PATTERN_STATE_CONTRADICTION
        ]
        assert len(contradictions) == 0


# ===================================================================
# Test class: Phantom step references
# ===================================================================


class TestPhantomStepReferences:
    """Test detection of references to non-existent steps."""

    def test_valid_step_reference_no_pattern(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Set up project",
                    step_number=1,
                    description="Initial setup",
                ),
                make_step(
                    "Continued from step 1",
                    step_number=2,
                    result_summary="Following up on step 1 work",
                ),
            ]
        )
        phantoms = [p for p in report.patterns if p.category == PATTERN_PHANTOM_STEP]
        assert len(phantoms) == 0

    def test_phantom_step_reference_detected(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Set up project",
                    step_number=1,
                ),
                make_step(
                    "Building on work from step 5",
                    step_number=2,
                    result_summary="Extending step 5 implementation",
                ),
            ]
        )
        phantoms = [p for p in report.patterns if p.category == PATTERN_PHANTOM_STEP]
        assert len(phantoms) >= 1
        assert "step 5" in phantoms[0].description.lower()

    def test_self_reference_not_flagged(self, tmp_path):
        """A step referencing its own number is not a phantom."""
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "This is step 1",
                    step_number=1,
                    result_summary="Step 1 complete",
                ),
            ]
        )
        phantoms = [p for p in report.patterns if p.category == PATTERN_PHANTOM_STEP]
        assert len(phantoms) == 0

    def test_no_step_references_no_pattern(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step("Did something", step_number=1),
                make_step("Did more", step_number=2),
            ]
        )
        phantoms = [p for p in report.patterns if p.category == PATTERN_PHANTOM_STEP]
        assert len(phantoms) == 0


# ===================================================================
# Test class: Severity classification
# ===================================================================


class TestSeverityClassification:
    """Test the severity classification logic."""

    def test_no_patterns_is_none(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check()
        assert report.severity == CONFUSION_SEVERITY_NONE

    def test_single_low_severity_pattern(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/maybe.py", "read")]
        )
        # 0.15 contribution -> low
        assert report.severity == CONFUSION_SEVERITY_LOW

    def test_medium_severity_from_multiple_patterns(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[
                make_file_record("src/a.py", "read"),
                make_file_record("src/b.py", "read"),
                make_file_record("src/c.py", "read"),
            ]
        )
        # 3 * 0.15 = 0.45 -> medium
        assert report.severity == CONFUSION_SEVERITY_MEDIUM

    def test_high_severity_from_state_contradiction(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Created the module",
                    success=True,
                    files=["src/phantom.py"],
                    step_number=1,
                ),
            ],
            file_records=[
                make_file_record("src/phantom.py", "created"),
            ],
        )
        # 0.5 (contradiction) + 0.4 (nonexistent from file_record) = 0.9 -> high
        assert report.severity == CONFUSION_SEVERITY_HIGH

    def test_mixed_patterns_medium_severity(self, tmp_path):
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/gone.py", "created")],
            step_records=[
                make_step(
                    "Extending step 99",
                    step_number=1,
                    result_summary="Based on step 99",
                ),
            ],
        )
        # 0.4 (nonexistent) + 0.25 (phantom step) = 0.65 -> high
        assert report.severity == CONFUSION_SEVERITY_HIGH


# ===================================================================
# Test class: TaskMemory integration
# ===================================================================


class TestTaskMemoryIntegration:
    """Test check_task() with TaskMemory objects."""

    def test_clean_task_no_confusion(self, tmp_path):
        create_file(tmp_path, "src/module.py", "class MyClass:\n    pass\n")
        detector = make_detector(tmp_path)
        task = TaskMemory(ticket_id="CMH-TEST", title="Test task")
        task.record_file("src/module.py", FileAction.CREATED, "New module")
        task.add_step(
            action="Created src/module.py",
            files_involved=["src/module.py"],
            success=True,
        )
        report = detector.check_task(task)
        assert not report.confusion_detected

    def test_task_with_missing_file(self, tmp_path):
        detector = make_detector(tmp_path)
        task = TaskMemory(ticket_id="CMH-TEST", title="Test task")
        task.record_file("src/phantom.py", FileAction.CREATED, "New module")
        report = detector.check_task(task)
        assert report.confusion_detected
        assert any(p.category == PATTERN_NONEXISTENT_FILE for p in report.patterns)

    def test_task_with_deleted_file_no_confusion(self, tmp_path):
        detector = make_detector(tmp_path)
        task = TaskMemory(ticket_id="CMH-TEST", title="Test task")
        task.record_file("src/old.py", FileAction.DELETED, "Removed old module")
        report = detector.check_task(task)
        assert not report.confusion_detected

    def test_task_multiple_issues(self, tmp_path):
        create_file(
            tmp_path,
            "src/real.py",
            "class RealClass:\n    pass\n",
        )
        detector = make_detector(tmp_path)
        task = TaskMemory(ticket_id="CMH-TEST", title="Test task")
        task.record_file("src/fake.py", FileAction.CREATED, "Fake file")
        task.add_step(
            action="Implemented WrongClassName class",
            files_involved=["src/real.py"],
            success=True,
        )
        report = detector.check_task(task)
        assert report.confusion_detected
        categories = {p.category for p in report.patterns}
        assert PATTERN_NONEXISTENT_FILE in categories
        assert PATTERN_WRONG_NAME in categories


# ===================================================================
# Test class: Report serialization
# ===================================================================


class TestReportSerialization:
    """Test ConfusionReport.to_dict()."""

    def test_empty_report_serialization(self):
        report = ConfusionReport()
        d = report.to_dict()
        assert d["confusion_detected"] is False
        assert d["severity"] == CONFUSION_SEVERITY_NONE
        assert d["patterns"] == []
        assert d["evidence"] == []
        assert isinstance(d["generated_at"], str)

    def test_report_with_patterns_serialization(self):
        report = ConfusionReport(
            confusion_detected=True,
            severity=CONFUSION_SEVERITY_HIGH,
            patterns=[
                ConfusionPattern(
                    category=PATTERN_NONEXISTENT_FILE,
                    description="File missing",
                    severity_contribution=0.4,
                    evidence="test evidence",
                    file_path="src/gone.py",
                    step_number=3,
                ),
            ],
            evidence=["test evidence"],
            total_files_checked=5,
            total_names_checked=2,
            total_steps_checked=3,
        )
        d = report.to_dict()
        assert d["confusion_detected"] is True
        assert d["severity"] == CONFUSION_SEVERITY_HIGH
        assert len(d["patterns"]) == 1
        assert d["patterns"][0]["category"] == PATTERN_NONEXISTENT_FILE
        assert d["patterns"][0]["file_path"] == "src/gone.py"
        assert d["patterns"][0]["step_number"] == 3
        assert d["total_files_checked"] == 5


# ===================================================================
# Test class: Utility functions
# ===================================================================


class TestUtilityFunctions:
    """Test module-level helper functions."""

    def test_levenshtein_identical(self):
        assert _levenshtein_distance("hello", "hello") == 0

    def test_levenshtein_one_char_diff(self):
        assert _levenshtein_distance("hello", "hallo") == 1

    def test_levenshtein_empty_strings(self):
        assert _levenshtein_distance("", "") == 0

    def test_levenshtein_one_empty(self):
        assert _levenshtein_distance("abc", "") == 3

    def test_levenshtein_insertions(self):
        assert _levenshtein_distance("abc", "abcd") == 1

    def test_levenshtein_case_sensitive(self):
        assert _levenshtein_distance("ABC", "abc") == 3

    def test_is_likely_class_name_valid(self):
        assert _is_likely_class_name("ConfusionDetector") is True
        assert _is_likely_class_name("TaskMemory") is True
        assert _is_likely_class_name("AlignmentChecker") is True

    def test_is_likely_class_name_short(self):
        assert _is_likely_class_name("Ab") is False
        assert _is_likely_class_name("Abc") is False

    def test_is_likely_class_name_all_upper(self):
        assert _is_likely_class_name("ALLCAPS") is False

    def test_is_likely_class_name_exceptions(self):
        assert _is_likely_class_name("ValueError") is False
        assert _is_likely_class_name("TypeError") is False

    def test_extract_name_references_class(self, tmp_path):
        detector = make_detector(tmp_path)
        names = detector._extract_name_references(
            "Implemented ConfusionDetector and TaskMemory"
        )
        assert "ConfusionDetector" in names
        assert "TaskMemory" in names

    def test_extract_name_references_function(self, tmp_path):
        detector = make_detector(tmp_path)
        names = detector._extract_name_references(
            "Called check_alignment() and record_step()"
        )
        assert "check_alignment" in names
        assert "record_step" in names

    def test_extract_defined_names_from_python(self, tmp_path):
        content = (
            "class MyClass:\n"
            "    pass\n\n"
            "def my_function():\n"
            "    pass\n\n"
            "async def async_handler():\n"
            "    pass\n"
        )
        names = ConfusionDetector._extract_defined_names(content)
        assert "MyClass" in names
        assert "my_function" in names
        assert "async_handler" in names

    def test_find_close_match_exact(self, tmp_path):
        detector = make_detector(tmp_path)
        result = detector._find_close_match("hello", {"hello", "world"})
        assert result == "hello"

    def test_find_close_match_typo(self, tmp_path):
        detector = make_detector(tmp_path)
        result = detector._find_close_match("helo", {"hello", "world"})
        assert result == "hello"

    def test_find_close_match_too_far(self, tmp_path):
        detector = make_detector(tmp_path)
        result = detector._find_close_match("xyz", {"hello", "world"})
        assert result is None

    def test_find_close_match_empty_candidates(self, tmp_path):
        detector = make_detector(tmp_path)
        result = detector._find_close_match("hello", set())
        assert result is None


# ===================================================================
# Test class: Path resolution
# ===================================================================


class TestPathResolution:
    """Test file path resolution behaviour."""

    def test_relative_path_resolved(self, tmp_path):
        detector = make_detector(tmp_path)
        resolved = detector._resolve_path("src/module.py")
        assert resolved == os.path.join(str(tmp_path), "src/module.py")

    def test_absolute_path_preserved(self, tmp_path):
        detector = make_detector(tmp_path)
        abs_path = "/absolute/path/to/file.py"
        resolved = detector._resolve_path(abs_path)
        assert resolved == abs_path


# ===================================================================
# Test class: Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unreadable_python_file_skipped(self, tmp_path):
        """Binary/unreadable files should not crash name detection."""
        bin_path = tmp_path / "src" / "binary.py"
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        bin_path.write_bytes(b"\x80\x81\x82\x83" * 100)
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                make_step(
                    "Updated SomeClass in binary file",
                    files=["src/binary.py"],
                )
            ]
        )
        # Should not crash; may or may not find patterns.
        assert isinstance(report, ConfusionReport)

    def test_very_large_step_count(self, tmp_path):
        """Detector handles many steps without performance issues."""
        create_file(tmp_path, "src/module.py", "def func():\n    pass\n")
        detector = make_detector(tmp_path)
        steps = [
            make_step(
                f"Step {i} operation",
                step_number=i,
                files=["src/module.py"],
            )
            for i in range(1, 51)
        ]
        report = detector.check(step_records=steps)
        assert report.total_steps_checked == 50

    def test_empty_file_path_in_record(self, tmp_path):
        """Empty file paths should be gracefully skipped."""
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[{"path": "", "action": "created"}]
        )
        # Empty path should not crash or produce meaningful pattern.
        assert isinstance(report, ConfusionReport)

    def test_special_characters_in_path(self, tmp_path):
        """Paths with special characters should be handled."""
        create_file(tmp_path, "src/my-module_v2.py", "# code")
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/my-module_v2.py", "modified")]
        )
        nonexist = [p for p in report.patterns if p.category == PATTERN_NONEXISTENT_FILE]
        assert len(nonexist) == 0

    def test_nested_directory_file(self, tmp_path):
        create_file(tmp_path, "src/deep/nested/module.py", "# deep")
        detector = make_detector(tmp_path)
        report = detector.check(
            file_records=[make_file_record("src/deep/nested/module.py", "created")]
        )
        nonexist = [p for p in report.patterns if p.category == PATTERN_NONEXISTENT_FILE]
        assert len(nonexist) == 0

    def test_step_with_none_result_summary(self, tmp_path):
        """Steps with None result_summary should not crash."""
        detector = make_detector(tmp_path)
        report = detector.check(
            step_records=[
                {
                    "action": "Did something",
                    "success": True,
                    "files_involved": [],
                    "step_number": 1,
                    "result_summary": None,
                    "tool_used": "",
                    "description": "",
                }
            ]
        )
        assert isinstance(report, ConfusionReport)
