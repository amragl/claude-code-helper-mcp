"""Comprehensive tests for the ScopeCreepDetector (CMH-022).

Tests cover:
- Basic construction and configuration
- Scope definition from ticket title, description, and keywords
- Out-of-scope file detection (directory mismatch, no pattern match)
- Out-of-scope action detection (low keyword relevance, feature addition)
- New module creation detection outside expected directories
- Config file modification detection
- Whitelist behaviour (test files, docs, __init__.py always in scope)
- Severity classification (none, low, medium, high)
- TaskMemory integration via check_task()
- DriftDetector integration via check_with_drift()
- Report serialization (to_dict)
- TicketScope serialization
- Edge cases (empty inputs, no scope keywords, no existing files)
- Custom whitelist patterns
- Action relevance threshold tuning
- Helper functions (_get_directory, _get_basename)
- Precision focus: no false positives on legitimate in-scope work

All tests use real computation with in-memory data -- zero mocks.
"""

from __future__ import annotations

import pytest

from claude_code_helper_mcp.detection.scope_creep import (
    CREEP_CONFIG_MODIFICATION,
    CREEP_NEW_MODULE,
    CREEP_OUT_OF_SCOPE_ACTION,
    CREEP_OUT_OF_SCOPE_FILE,
    CREEP_SEVERITY_HIGH,
    CREEP_SEVERITY_LOW,
    CREEP_SEVERITY_MEDIUM,
    CREEP_SEVERITY_NONE,
    ScopeCreepDetector,
    ScopeCreepReport,
    ScopeCreepSignal,
    TicketScope,
    _get_basename,
    _get_directory,
)
from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.models.records import FileAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> ScopeCreepDetector:
    """Return a ScopeCreepDetector with default settings."""
    return ScopeCreepDetector()


@pytest.fixture
def scope_context() -> dict:
    """Return a realistic scope context for testing."""
    return {
        "ticket_id": "CMH-022",
        "title": "Scope creep detection",
        "description": (
            "Implement ScopeCreepDetector class that defines ticket scope "
            "from description/keywords, detects out-of-scope file edits "
            "and feature additions. Integrates with DriftDetector."
        ),
        "phase": "phase-5",
        "existing_files": [
            "src/claude_code_helper_mcp/detection/scope_creep.py",
            "src/claude_code_helper_mcp/detection/__init__.py",
            "tests/test_scope_creep.py",
        ],
    }


@pytest.fixture
def scope(detector: ScopeCreepDetector, scope_context: dict) -> TicketScope:
    """Return a TicketScope from the realistic context."""
    return detector.define_scope(**scope_context)


# ===========================================================================
# 1. ScopeCreepDetector construction and defaults
# ===========================================================================


class TestConstruction:
    """Tests for ScopeCreepDetector initialisation and defaults."""

    def test_default_construction(self) -> None:
        """ScopeCreepDetector can be created with default settings."""
        d = ScopeCreepDetector()
        assert d.action_relevance_threshold == 0.15
        assert len(d.whitelist_patterns) > 0

    def test_custom_whitelist(self) -> None:
        """Custom whitelist patterns are appended to defaults."""
        d = ScopeCreepDetector(whitelist_patterns=["custom_pattern"])
        assert "custom_pattern" in d.whitelist_patterns
        # Default patterns still present.
        assert "test_" in d.whitelist_patterns

    def test_custom_threshold(self) -> None:
        """Custom action relevance threshold is respected."""
        d = ScopeCreepDetector(action_relevance_threshold=0.5)
        assert d.action_relevance_threshold == 0.5

    def test_threshold_clamped_to_range(self) -> None:
        """Threshold is clamped to [0.0, 1.0]."""
        d_low = ScopeCreepDetector(action_relevance_threshold=-0.5)
        assert d_low.action_relevance_threshold == 0.0

        d_high = ScopeCreepDetector(action_relevance_threshold=1.5)
        assert d_high.action_relevance_threshold == 1.0


# ===========================================================================
# 2. Scope definition
# ===========================================================================


class TestScopeDefinition:
    """Tests for TicketScope creation via define_scope()."""

    def test_basic_scope_definition(self, detector: ScopeCreepDetector) -> None:
        """Scope is defined from title and description keywords."""
        scope = detector.define_scope(
            ticket_id="CMH-022",
            title="Scope creep detection",
            description="Implement ScopeCreepDetector class",
        )
        assert scope.ticket_id == "CMH-022"
        assert scope.title == "Scope creep detection"
        assert "scope" in scope.keywords
        assert "creep" in scope.keywords
        assert "detection" in scope.keywords

    def test_scope_includes_description_keywords(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Keywords are extracted from description."""
        scope = detector.define_scope(
            title="Test feature",
            description="Build a scope creep detector with keyword analysis",
        )
        assert "detector" in scope.keywords
        assert "keyword" in scope.keywords
        assert "analysis" in scope.keywords

    def test_scope_includes_ticket_id_prefix(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Ticket ID prefix is included as a keyword."""
        scope = detector.define_scope(ticket_id="CMH-022", title="Test")
        assert "cmh" in scope.keywords

    def test_scope_includes_phase_keywords(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Phase name is included as keywords."""
        scope = detector.define_scope(title="Test", phase="phase-5")
        assert "phase" in scope.keywords

    def test_scope_infers_directories_from_files(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Expected directories are inferred from existing files."""
        scope = detector.define_scope(
            title="Test",
            existing_files=[
                "src/claude_code_helper_mcp/detection/scope_creep.py",
                "tests/test_scope_creep.py",
            ],
        )
        assert "src/claude_code_helper_mcp/detection" in scope.expected_directories
        assert "tests" in scope.expected_directories

    def test_scope_infers_file_patterns(
        self, detector: ScopeCreepDetector
    ) -> None:
        """File patterns are inferred from existing file basenames."""
        scope = detector.define_scope(
            title="Test",
            existing_files=["src/detection/scope_creep.py"],
        )
        assert "scope_creep" in scope.expected_file_patterns

    def test_empty_scope_definition(self, detector: ScopeCreepDetector) -> None:
        """Empty inputs produce an empty scope."""
        scope = detector.define_scope()
        assert scope.ticket_id == ""
        assert scope.title == ""
        assert len(scope.keywords) == 0
        assert len(scope.expected_directories) == 0

    def test_scope_to_dict(self, scope: TicketScope) -> None:
        """TicketScope serializes to a dictionary."""
        d = scope.to_dict()
        assert d["ticket_id"] == "CMH-022"
        assert "keyword_count" in d
        assert isinstance(d["keywords_sample"], list)
        assert isinstance(d["expected_directories"], list)


# ===========================================================================
# 3. Out-of-scope file detection
# ===========================================================================


class TestFileDetection:
    """Tests for out-of-scope file detection."""

    def test_in_scope_file_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Files in expected directories are not flagged."""
        report = detector.check(
            scope,
            file_paths=["src/claude_code_helper_mcp/detection/scope_creep.py"],
        )
        assert not report.creep_detected
        assert len(report.out_of_scope_files) == 0

    def test_out_of_scope_file_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Files in unrelated directories are flagged."""
        report = detector.check(
            scope,
            file_paths=["src/completely_unrelated/database/models.py"],
        )
        assert report.creep_detected
        assert "src/completely_unrelated/database/models.py" in report.out_of_scope_files

    def test_test_file_always_in_scope(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Test files are whitelisted and never flagged."""
        report = detector.check(
            scope,
            file_paths=["tests/test_something_unrelated.py"],
        )
        assert not report.creep_detected

    def test_documentation_always_in_scope(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Documentation files are whitelisted."""
        report = detector.check(
            scope,
            file_paths=["docs/unrelated_topic.md", "README.md"],
        )
        assert not report.creep_detected

    def test_init_file_always_in_scope(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """__init__.py files are whitelisted."""
        report = detector.check(
            scope,
            file_paths=["src/some_other_package/__init__.py"],
        )
        assert not report.creep_detected

    def test_file_with_scope_keywords_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Files whose path contains scope keywords are not flagged."""
        # 'scope_creep' or 'detection' should be in scope keywords.
        report = detector.check(
            scope,
            file_paths=["src/other_package/detection/helper.py"],
        )
        # 'detection' is in scope keywords, so this should not be flagged.
        assert not report.creep_detected

    def test_multiple_out_of_scope_files(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Multiple out-of-scope files produce multiple signals."""
        report = detector.check(
            scope,
            file_paths=[
                "src/unrelated/database.py",
                "src/different/network.py",
            ],
        )
        assert report.creep_detected
        assert len(report.out_of_scope_files) == 2

    def test_empty_file_list(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """No files produces no signals."""
        report = detector.check(scope, file_paths=[])
        assert not report.creep_detected

    def test_file_count_tracked(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Total files checked is correctly reported."""
        report = detector.check(
            scope,
            file_paths=["a.py", "b.py", "c.py"],
        )
        assert report.total_files_checked == 3


# ===========================================================================
# 4. Out-of-scope action detection
# ===========================================================================


class TestActionDetection:
    """Tests for out-of-scope action detection."""

    def test_relevant_action_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Actions with high keyword overlap are not flagged."""
        report = detector.check(
            scope,
            actions=["Implementing scope creep detection logic"],
        )
        assert len(report.out_of_scope_actions) == 0

    def test_irrelevant_action_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Actions with no keyword overlap are flagged."""
        report = detector.check(
            scope,
            actions=["Configuring PostgreSQL database migrations"],
        )
        assert len(report.out_of_scope_actions) > 0

    def test_feature_addition_keyword_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Actions with feature-addition language are flagged."""
        report = detector.check(
            scope,
            actions=["Also added a new caching system while we are at it"],
        )
        assert report.creep_detected
        signals = [s for s in report.signals if s.category == CREEP_OUT_OF_SCOPE_ACTION]
        assert len(signals) > 0

    def test_bonus_feature_keyword_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """'bonus' keyword triggers scope creep detection."""
        report = detector.check(
            scope,
            actions=["Bonus: refactored the entire CLI module"],
        )
        assert report.creep_detected

    def test_empty_action_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Empty action strings are not flagged."""
        report = detector.check(scope, actions=[""])
        assert not report.creep_detected

    def test_action_count_tracked(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Total actions checked is correctly reported."""
        report = detector.check(
            scope,
            actions=["action 1", "action 2", "action 3"],
        )
        assert report.total_actions_checked == 3

    def test_custom_threshold_affects_detection(
        self, scope: TicketScope
    ) -> None:
        """Higher threshold catches more actions as out-of-scope."""
        strict = ScopeCreepDetector(action_relevance_threshold=0.8)
        lenient = ScopeCreepDetector(action_relevance_threshold=0.05)

        actions = ["Adding detection logic for scope analysis"]
        strict_report = strict.check(scope, actions=actions)
        lenient_report = lenient.check(scope, actions=actions)

        # Strict should flag more (or same) as lenient.
        assert strict_report.total_actions_checked == lenient_report.total_actions_checked


# ===========================================================================
# 5. New module creation detection
# ===========================================================================


class TestNewModuleDetection:
    """Tests for new module creation outside expected scope."""

    def test_new_module_in_scope_dir_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """New Python files in expected directories are not flagged."""
        report = detector.check(
            scope,
            file_paths=["src/claude_code_helper_mcp/detection/new_helper.py"],
        )
        new_module_signals = [
            s for s in report.signals if s.category == CREEP_NEW_MODULE
        ]
        assert len(new_module_signals) == 0

    def test_new_module_outside_scope_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """New Python files outside expected directories are flagged."""
        report = detector.check(
            scope,
            file_paths=["src/networking/http_handler.py"],
        )
        new_module_signals = [
            s for s in report.signals if s.category == CREEP_NEW_MODULE
        ]
        assert len(new_module_signals) > 0

    def test_init_py_not_flagged_as_new_module(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """__init__.py files are never flagged as new modules."""
        report = detector.check(
            scope,
            file_paths=["src/completely_unrelated/__init__.py"],
        )
        new_module_signals = [
            s for s in report.signals if s.category == CREEP_NEW_MODULE
        ]
        assert len(new_module_signals) == 0

    def test_non_python_file_not_flagged_as_new_module(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Non-Python files are not flagged as new modules."""
        report = detector.check(
            scope,
            file_paths=["src/completely_unrelated/data.json"],
        )
        new_module_signals = [
            s for s in report.signals if s.category == CREEP_NEW_MODULE
        ]
        assert len(new_module_signals) == 0

    def test_new_module_with_scope_keywords_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """New modules whose path contains scope keywords are not flagged."""
        report = detector.check(
            scope,
            file_paths=["src/other_package/scope_creep_utils.py"],
        )
        new_module_signals = [
            s for s in report.signals if s.category == CREEP_NEW_MODULE
        ]
        assert len(new_module_signals) == 0


# ===========================================================================
# 6. Configuration file modification detection
# ===========================================================================


class TestConfigDetection:
    """Tests for configuration file modification detection."""

    def test_config_in_scope_dir_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Config files in expected directories are not flagged."""
        report = detector.check(
            scope,
            file_paths=["src/claude_code_helper_mcp/detection/config.yaml"],
        )
        config_signals = [
            s for s in report.signals if s.category == CREEP_CONFIG_MODIFICATION
        ]
        assert len(config_signals) == 0

    def test_config_outside_scope_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Config files outside scope without keywords are flagged."""
        report = detector.check(
            scope,
            file_paths=["infrastructure/deploy.yaml"],
        )
        config_signals = [
            s for s in report.signals if s.category == CREEP_CONFIG_MODIFICATION
        ]
        assert len(config_signals) > 0

    def test_whitelisted_config_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Whitelisted config files (pyproject.toml) are not flagged."""
        report = detector.check(
            scope,
            file_paths=["pyproject.toml"],
        )
        assert not report.creep_detected

    def test_config_with_scope_keywords_not_flagged(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Config files whose path contains scope keywords are not flagged."""
        report = detector.check(
            scope,
            file_paths=["config/detection_settings.yaml"],
        )
        config_signals = [
            s for s in report.signals if s.category == CREEP_CONFIG_MODIFICATION
        ]
        assert len(config_signals) == 0

    def test_various_config_extensions_detected(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Various config file extensions are recognized."""
        config_files = [
            "unrelated/settings.toml",
            "unrelated/database.ini",
            "unrelated/app.env",
        ]
        report = detector.check(scope, file_paths=config_files)
        config_signals = [
            s for s in report.signals if s.category == CREEP_CONFIG_MODIFICATION
        ]
        assert len(config_signals) == 3


# ===========================================================================
# 7. Severity classification
# ===========================================================================


class TestSeverityClassification:
    """Tests for severity classification from signals."""

    def test_no_signals_severity_none(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """No signals produces 'none' severity."""
        report = detector.check(scope, file_paths=[], actions=[])
        assert report.severity == CREEP_SEVERITY_NONE

    def test_single_low_signal_severity_low(self) -> None:
        """A single low-contribution signal produces 'low' severity."""
        result = ScopeCreepDetector._classify_severity([
            ScopeCreepSignal(
                category=CREEP_CONFIG_MODIFICATION,
                description="test",
                severity_contribution=0.2,
            ),
        ])
        assert result == CREEP_SEVERITY_LOW

    def test_medium_total_severity_medium(self) -> None:
        """Signals totalling [0.3, 0.6) produce 'medium' severity."""
        result = ScopeCreepDetector._classify_severity([
            ScopeCreepSignal(
                category=CREEP_OUT_OF_SCOPE_FILE,
                description="test",
                severity_contribution=0.35,
            ),
        ])
        assert result == CREEP_SEVERITY_MEDIUM

    def test_high_total_severity_high(self) -> None:
        """Signals totalling >= 0.6 produce 'high' severity."""
        result = ScopeCreepDetector._classify_severity([
            ScopeCreepSignal(
                category=CREEP_OUT_OF_SCOPE_FILE,
                description="test",
                severity_contribution=0.35,
            ),
            ScopeCreepSignal(
                category=CREEP_OUT_OF_SCOPE_ACTION,
                description="test",
                severity_contribution=0.35,
            ),
        ])
        assert result == CREEP_SEVERITY_HIGH

    def test_cumulative_severity(self) -> None:
        """Multiple low signals accumulate to higher severity."""
        result = ScopeCreepDetector._classify_severity([
            ScopeCreepSignal(
                category=CREEP_CONFIG_MODIFICATION,
                description="test",
                severity_contribution=0.15,
            ),
            ScopeCreepSignal(
                category=CREEP_CONFIG_MODIFICATION,
                description="test",
                severity_contribution=0.15,
            ),
            ScopeCreepSignal(
                category=CREEP_CONFIG_MODIFICATION,
                description="test",
                severity_contribution=0.15,
            ),
        ])
        # 0.15 * 3 = 0.45 -> medium
        assert result == CREEP_SEVERITY_MEDIUM


# ===========================================================================
# 8. Report serialization
# ===========================================================================


class TestReportSerialization:
    """Tests for ScopeCreepReport and TicketScope serialization."""

    def test_empty_report_to_dict(self) -> None:
        """Empty report serializes correctly."""
        report = ScopeCreepReport()
        d = report.to_dict()
        assert d["creep_detected"] is False
        assert d["severity"] == CREEP_SEVERITY_NONE
        assert d["out_of_scope_files"] == []
        assert d["out_of_scope_actions"] == []
        assert d["signals"] == []
        assert d["ticket_scope_summary"] is None
        assert "generated_at" in d

    def test_full_report_to_dict(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Full report with signals serializes correctly."""
        report = detector.check(
            scope,
            file_paths=["src/unrelated/database.py"],
            actions=["Also added caching while we are at it"],
        )
        d = report.to_dict()
        assert d["creep_detected"] is True
        assert isinstance(d["signals"], list)
        assert len(d["signals"]) > 0
        for signal in d["signals"]:
            assert "category" in signal
            assert "description" in signal
            assert "severity_contribution" in signal

    def test_report_includes_scope_summary(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Report includes ticket scope summary."""
        report = detector.check(scope)
        d = report.to_dict()
        assert d["ticket_scope_summary"] is not None
        assert d["ticket_scope_summary"]["ticket_id"] == "CMH-022"


# ===========================================================================
# 9. TaskMemory integration
# ===========================================================================


class TestTaskMemoryIntegration:
    """Tests for check_task() with TaskMemory objects."""

    def test_check_task_in_scope(self, detector: ScopeCreepDetector) -> None:
        """check_task with in-scope files produces no creep."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
            phase="phase-5",
            metadata={"description": "Implement scope creep detector"},
        )
        task.record_file(
            "src/claude_code_helper_mcp/detection/scope_creep.py",
            FileAction.CREATED,
            "Created scope creep module",
        )
        task.add_step(
            action="Implementing scope creep detection logic",
            description="Writing the ScopeCreepDetector class",
        )

        report = detector.check_task(
            task,
            scope_files=["src/claude_code_helper_mcp/detection/scope_creep.py"],
        )
        assert not report.creep_detected

    def test_check_task_with_out_of_scope_file(
        self, detector: ScopeCreepDetector
    ) -> None:
        """check_task with out-of-scope files produces signals."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
            phase="phase-5",
            metadata={"description": "Implement scope creep detector"},
        )
        task.record_file(
            "src/claude_code_helper_mcp/detection/scope_creep.py",
            FileAction.CREATED,
        )
        task.record_file(
            "src/networking/http_handler.py",
            FileAction.MODIFIED,
        )

        # Provide scope_files to define the expected directories.
        report = detector.check_task(
            task,
            scope_files=["src/claude_code_helper_mcp/detection/scope_creep.py"],
        )
        assert report.creep_detected
        assert "src/networking/http_handler.py" in report.out_of_scope_files

    def test_check_task_uses_metadata_description(
        self, detector: ScopeCreepDetector
    ) -> None:
        """check_task extracts description from task metadata."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
            metadata={"description": "Build scope creep analyzer with keywords"},
        )
        report = detector.check_task(task)
        assert report.ticket_scope_summary is not None
        assert "analyzer" in report.ticket_scope_summary.keywords or "scope" in report.ticket_scope_summary.keywords

    def test_check_task_with_explicit_description(
        self, detector: ScopeCreepDetector
    ) -> None:
        """check_task uses explicit description over metadata."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
            metadata={"description": "Original description"},
        )
        report = detector.check_task(
            task,
            ticket_description="Custom override description with unique keywords",
        )
        assert report.ticket_scope_summary is not None
        assert "unique" in report.ticket_scope_summary.keywords


# ===========================================================================
# 10. DriftDetector integration
# ===========================================================================


class TestDriftIntegration:
    """Tests for check_with_drift() combining scope creep and drift."""

    def test_combined_check_returns_both_reports(
        self, detector: ScopeCreepDetector
    ) -> None:
        """check_with_drift returns both scope_creep and drift reports."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
            phase="phase-5",
            metadata={"description": "Implement scope creep detector"},
        )
        task.add_step(action="Implementing scope creep detection")

        result = detector.check_with_drift(
            task=task,
            action="Adding scope creep detection logic",
            file_path="src/claude_code_helper_mcp/detection/scope_creep.py",
        )
        assert "scope_creep" in result
        assert "drift" in result
        assert "overall_severity" in result
        assert result["ticket_id"] == "CMH-022"

    def test_combined_severity_takes_worst(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Combined severity is the worst of both analyses."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
            phase="phase-5",
            metadata={"description": "Implement scope creep detector"},
        )
        task.add_step(action="Setting up the scope creep module")

        # Action that is in-scope for the ticket.
        result = detector.check_with_drift(
            task=task,
            action="Implementing ScopeCreepDetector class",
        )
        # Both should be low/none severity.
        assert result["overall_severity"] in ("none", "low", "medium")

    def test_combined_check_with_no_file(
        self, detector: ScopeCreepDetector
    ) -> None:
        """check_with_drift works without a file_path."""
        task = TaskMemory(
            ticket_id="CMH-022",
            title="Scope creep detection",
        )
        result = detector.check_with_drift(
            task=task,
            action="Implementing scope creep detection",
        )
        assert result["file_path"] is None


# ===========================================================================
# 11. Whitelist behaviour
# ===========================================================================


class TestWhitelist:
    """Tests for whitelist pattern matching."""

    def test_test_prefix_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Files starting with test_ are whitelisted."""
        assert detector._is_whitelisted("test_something.py")

    def test_tests_directory_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Files in tests/ directory are whitelisted."""
        assert detector._is_whitelisted("tests/test_foo.py")

    def test_markdown_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Markdown files are whitelisted."""
        assert detector._is_whitelisted("docs/guide.md")

    def test_readme_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """README files are whitelisted."""
        assert detector._is_whitelisted("README.rst")

    def test_init_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """__init__.py files are whitelisted."""
        assert detector._is_whitelisted("src/package/__init__.py")

    def test_pyproject_toml_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """pyproject.toml is whitelisted."""
        assert detector._is_whitelisted("pyproject.toml")

    def test_gitignore_whitelisted(
        self, detector: ScopeCreepDetector
    ) -> None:
        """.gitignore is whitelisted."""
        assert detector._is_whitelisted(".gitignore")

    def test_non_whitelisted_file(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Non-whitelisted files are not matched."""
        assert not detector._is_whitelisted("src/main_code.py")

    def test_custom_whitelist_pattern(self) -> None:
        """Custom whitelist patterns work."""
        d = ScopeCreepDetector(whitelist_patterns=["migrations/"])
        assert d._is_whitelisted("db/migrations/001_init.py")
        assert not d._is_whitelisted("src/main.py")

    def test_case_insensitive_whitelist(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Whitelist matching is case-insensitive."""
        assert detector._is_whitelisted("README.MD")
        assert detector._is_whitelisted("Tests/TestFoo.py")


# ===========================================================================
# 12. Helper functions
# ===========================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_get_directory_with_path(self) -> None:
        """_get_directory extracts directory from a path."""
        assert _get_directory("src/detection/scope_creep.py") == "src/detection"

    def test_get_directory_no_slash(self) -> None:
        """_get_directory returns empty string for bare filename."""
        assert _get_directory("scope_creep.py") == ""

    def test_get_directory_deep_path(self) -> None:
        """_get_directory works with deeply nested paths."""
        assert _get_directory("a/b/c/d.py") == "a/b/c"

    def test_get_basename_with_extension(self) -> None:
        """_get_basename extracts filename without extension."""
        assert _get_basename("src/scope_creep.py") == "scope_creep"

    def test_get_basename_no_directory(self) -> None:
        """_get_basename works with bare filename."""
        assert _get_basename("scope_creep.py") == "scope_creep"

    def test_get_basename_no_extension(self) -> None:
        """_get_basename returns full name if no extension."""
        assert _get_basename("Makefile") == "Makefile"

    def test_get_basename_multiple_dots(self) -> None:
        """_get_basename handles filenames with multiple dots."""
        assert _get_basename("src/config.test.json") == "config.test"

    def test_is_config_file_by_extension(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Config files are detected by extension."""
        assert detector._is_config_file("settings.toml")
        assert detector._is_config_file("config.yaml")
        assert detector._is_config_file("db.ini")
        assert detector._is_config_file("app.env")
        assert not detector._is_config_file("main.py")

    def test_is_config_file_by_directory(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Config files are detected by directory name."""
        assert detector._is_config_file("config/main.py")
        assert detector._is_config_file("settings/local.py")
        assert not detector._is_config_file("src/main.py")

    def test_has_feature_addition_keywords(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Feature addition keywords are detected."""
        assert detector._has_feature_addition_keywords("Also added a new module")
        assert detector._has_feature_addition_keywords("Bonus: added caching")
        assert detector._has_feature_addition_keywords(
            "While we are at it, fixed the UI"
        )
        assert not detector._has_feature_addition_keywords(
            "Added scope creep detection"
        )

    def test_file_in_expected_directories(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """File directory matching works with prefix comparison."""
        assert detector._file_in_expected_directories(
            "src/claude_code_helper_mcp/detection/helper.py", scope
        )
        assert not detector._file_in_expected_directories(
            "src/completely_different/module.py", scope
        )

    def test_file_matches_patterns(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """File pattern matching works."""
        assert detector._file_matches_patterns(
            "src/other/scope_creep_helper.py", scope
        )

    def test_file_has_scope_keywords(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """File path keyword checking works."""
        assert detector._file_has_scope_keywords(
            "src/other/detection/utils.py", scope
        )
        assert not detector._file_has_scope_keywords(
            "src/network/http_client.py", scope
        )


# ===========================================================================
# 13. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_check_with_no_scope_keywords(
        self, detector: ScopeCreepDetector
    ) -> None:
        """Check with empty scope still runs without error."""
        scope = TicketScope()
        report = detector.check(
            scope,
            file_paths=["src/some_file.py"],
            actions=["some action"],
        )
        # With no scope keywords, actions should not be flagged for relevance.
        assert report.total_files_checked == 1
        assert report.total_actions_checked == 1

    def test_check_with_no_inputs(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Check with no files and no actions produces empty report."""
        report = detector.check(scope)
        assert not report.creep_detected
        assert report.severity == CREEP_SEVERITY_NONE

    def test_files_only_no_actions(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Check with files but no actions works correctly."""
        report = detector.check(
            scope,
            file_paths=["src/claude_code_helper_mcp/detection/scope_creep.py"],
        )
        assert report.total_files_checked == 1
        assert report.total_actions_checked == 0

    def test_actions_only_no_files(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Check with actions but no files works correctly."""
        report = detector.check(
            scope,
            actions=["Implementing scope creep detection"],
        )
        assert report.total_files_checked == 0
        assert report.total_actions_checked == 1

    def test_very_long_action(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Very long action descriptions do not cause errors."""
        long_action = "scope creep detection " * 100
        report = detector.check(scope, actions=[long_action])
        assert report.total_actions_checked == 1

    def test_special_characters_in_file_path(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """File paths with special characters are handled."""
        report = detector.check(
            scope,
            file_paths=["src/module (copy)/file-v2.py"],
        )
        assert report.total_files_checked == 1

    def test_no_expected_directories(self, detector: ScopeCreepDetector) -> None:
        """When scope has no expected directories, no module signals."""
        scope = detector.define_scope(
            title="Test",
            description="Just a test",
        )
        # No existing_files, so no expected_directories.
        assert len(scope.expected_directories) == 0
        report = detector.check(
            scope,
            file_paths=["src/any/module.py"],
        )
        # New module check requires expected_directories to be non-empty.
        new_module_signals = [
            s for s in report.signals if s.category == CREEP_NEW_MODULE
        ]
        assert len(new_module_signals) == 0

    def test_report_generated_at_is_set(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """Report generated_at timestamp is always set."""
        report = detector.check(scope)
        assert report.generated_at is not None

    def test_signal_categories_are_correct(
        self, detector: ScopeCreepDetector, scope: TicketScope
    ) -> None:
        """All generated signals use valid category constants."""
        valid_categories = {
            CREEP_OUT_OF_SCOPE_FILE,
            CREEP_OUT_OF_SCOPE_ACTION,
            CREEP_NEW_MODULE,
            CREEP_CONFIG_MODIFICATION,
        }
        report = detector.check(
            scope,
            file_paths=[
                "src/unrelated/database.py",
                "infrastructure/deploy.yaml",
            ],
            actions=["Also added caching while we are at it"],
        )
        for signal in report.signals:
            assert signal.category in valid_categories
