"""Scope creep detection engine for identifying out-of-scope work.

The ScopeCreepDetector analyses actions and file modifications against the
active ticket's defined scope to detect when work expands beyond the ticket's
requirements.  It flags:

- File edits in unrelated packages or directories
- Actions that introduce entirely new features not in the ticket description
- Modifications to configuration files not required by the task
- Creation of new modules that go beyond the ticket requirements

This module provides:
- Ticket scope definition from title, description, keywords, and file patterns
- Out-of-scope file detection with configurable whitelist
- Out-of-scope action detection with keyword relevance analysis
- ScopeCreepReport with severity, evidence, and scope summary
- Integration with DriftDetector for combined drift+creep analysis
- Integration with TaskMemory for automatic analysis of recorded data

Design decisions:
- All analysis is local and deterministic (no external API calls).
- ScopeCreepDetector is stateless per check -- call check() or check_task()
  each time.  No session state is accumulated.
- Whitelist: test files, documentation, and __init__.py files are always
  considered in-scope to prevent false positives on necessary housekeeping.
- Scope inference uses keyword extraction consistent with AlignmentChecker
  and DriftDetector for uniformity across the detection subsystem.
- Reasonable scope expansion is tolerated: fixing imports when adding a
  module, updating __init__.py re-exports, and touching adjacent test files
  are not flagged.

Depends on:
- CMH-016: StateReader / pipeline state file readers (for backlog ticket context)
- CMH-019: DriftDetector (for combined drift+creep integration)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from claude_code_helper_mcp.detection.alignment import AlignmentChecker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Severity levels for scope creep detection.
CREEP_SEVERITY_NONE = "none"
CREEP_SEVERITY_LOW = "low"
CREEP_SEVERITY_MEDIUM = "medium"
CREEP_SEVERITY_HIGH = "high"

# Category constants for scope creep signals.
CREEP_OUT_OF_SCOPE_FILE = "out_of_scope_file"
CREEP_OUT_OF_SCOPE_ACTION = "out_of_scope_action"
CREEP_NEW_MODULE = "new_module"
CREEP_CONFIG_MODIFICATION = "config_modification"

# File patterns that are always considered in-scope (whitelist).
# These are files that commonly need updating during any ticket work.
_ALWAYS_IN_SCOPE_PATTERNS: tuple[str, ...] = (
    # Test files are always in scope.
    "test_",
    "tests/",
    "test/",
    # Documentation is always in scope.
    ".md",
    "README",
    "CHANGELOG",
    "LICENSE",
    # Package init files are always in scope (re-exports, imports).
    "__init__",
    # Type stubs.
    ".pyi",
    # Configuration files for the project itself.
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    ".gitignore",
)

# File extensions that indicate configuration files.
_CONFIG_EXTENSIONS: frozenset[str] = frozenset({
    ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".env",
    ".conf", ".config",
})

# Directories that indicate configuration areas.
_CONFIG_DIRECTORIES: frozenset[str] = frozenset({
    "config", "configs", ".config", "settings", "conf",
    ".agent-forge", ".claude",
})

# Keywords that strongly indicate feature additions beyond ticket scope.
_FEATURE_ADDITION_KEYWORDS: frozenset[str] = frozenset({
    "new feature", "bonus", "extra", "additionally", "while we are at it",
    "also added", "refactored", "reorganized", "migrated",
    "unrelated fix", "quick fix for",
})

# Minimum keyword overlap ratio for an action to be considered in-scope.
_MIN_ACTION_RELEVANCE = 0.15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScopeCreepSignal:
    """A single scope creep signal detected during analysis.

    Attributes:
        category: The type of scope creep (e.g., 'out_of_scope_file',
            'out_of_scope_action', 'new_module', 'config_modification').
        description: Human-readable description of the scope creep signal.
        severity_contribution: Float in [0.0, 1.0] indicating how much this
            signal contributes to overall scope creep severity.
        evidence: Specific evidence supporting the detection.
        file_path: The file involved (if applicable).
    """

    category: str
    description: str
    severity_contribution: float
    evidence: str = ""
    file_path: str = ""


@dataclass
class TicketScope:
    """Defines the expected scope of a ticket for comparison.

    Attributes:
        ticket_id: The ticket identifier (e.g., 'CMH-022').
        title: The ticket title.
        description: The ticket description.
        keywords: Extracted keywords defining the ticket's domain.
        expected_directories: Directories expected to be touched.
        expected_file_patterns: File name patterns expected to be touched.
        phase: The roadmap phase this ticket belongs to.
    """

    ticket_id: str = ""
    title: str = ""
    description: str = ""
    keywords: set[str] = field(default_factory=set)
    expected_directories: set[str] = field(default_factory=set)
    expected_file_patterns: set[str] = field(default_factory=set)
    phase: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "description": self.description[:200] + ("..." if len(self.description) > 200 else ""),
            "keyword_count": len(self.keywords),
            "keywords_sample": sorted(self.keywords)[:15],
            "expected_directories": sorted(self.expected_directories),
            "expected_file_patterns": sorted(self.expected_file_patterns),
            "phase": self.phase,
        }


@dataclass
class ScopeCreepReport:
    """Result of a scope creep detection check.

    Attributes:
        creep_detected: Boolean. True when at least one scope creep signal
            is found.
        severity: One of 'none', 'low', 'medium', 'high'.
        out_of_scope_files: List of file paths that appear outside the
            ticket's expected scope.
        out_of_scope_actions: List of action descriptions that appear
            unrelated to the ticket.
        signals: List of ScopeCreepSignal objects describing detected issues.
        ticket_scope_summary: Summary of what the ticket is supposed to cover.
        total_files_checked: Number of file references analysed.
        total_actions_checked: Number of actions analysed.
        generated_at: UTC timestamp when this report was generated.
    """

    creep_detected: bool = False
    severity: str = CREEP_SEVERITY_NONE
    out_of_scope_files: list[str] = field(default_factory=list)
    out_of_scope_actions: list[str] = field(default_factory=list)
    signals: list[ScopeCreepSignal] = field(default_factory=list)
    ticket_scope_summary: Optional[TicketScope] = None
    total_files_checked: int = 0
    total_actions_checked: int = 0
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "creep_detected": self.creep_detected,
            "severity": self.severity,
            "out_of_scope_files": list(self.out_of_scope_files),
            "out_of_scope_actions": list(self.out_of_scope_actions),
            "signals": [
                {
                    "category": s.category,
                    "description": s.description,
                    "severity_contribution": round(s.severity_contribution, 3),
                    "evidence": s.evidence,
                    "file_path": s.file_path,
                }
                for s in self.signals
            ],
            "ticket_scope_summary": (
                self.ticket_scope_summary.to_dict()
                if self.ticket_scope_summary
                else None
            ),
            "total_files_checked": self.total_files_checked,
            "total_actions_checked": self.total_actions_checked,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# ScopeCreepDetector
# ---------------------------------------------------------------------------


class ScopeCreepDetector:
    """Detects scope creep by comparing actions and files against ticket scope.

    The detector defines the expected scope of a ticket from its title,
    description, and keywords, then checks whether recorded files and
    actions fall within that scope.  Out-of-scope items are flagged with
    severity levels based on how far they diverge from the ticket's intent.

    Parameters
    ----------
    whitelist_patterns:
        Additional file patterns to always consider in-scope (beyond the
        built-in defaults).  Each pattern is checked as a substring match
        against file paths.
    action_relevance_threshold:
        Minimum keyword overlap ratio for an action to be considered
        in-scope.  Default 0.15.  Lower values are more permissive.
    """

    def __init__(
        self,
        whitelist_patterns: Optional[list[str]] = None,
        action_relevance_threshold: float = _MIN_ACTION_RELEVANCE,
    ) -> None:
        self._whitelist = list(_ALWAYS_IN_SCOPE_PATTERNS)
        if whitelist_patterns:
            self._whitelist.extend(whitelist_patterns)
        self._action_threshold = max(0.0, min(1.0, action_relevance_threshold))

    @property
    def whitelist_patterns(self) -> list[str]:
        """Return a copy of the current whitelist patterns."""
        return list(self._whitelist)

    @property
    def action_relevance_threshold(self) -> float:
        """Return the current action relevance threshold."""
        return self._action_threshold

    def define_scope(
        self,
        ticket_id: str = "",
        title: str = "",
        description: str = "",
        phase: str = "",
        existing_files: Optional[list[str]] = None,
    ) -> TicketScope:
        """Define the expected scope for a ticket.

        Extracts keywords from the title and description, infers expected
        directories from existing task files, and builds a TicketScope
        object that will be used as the reference for scope creep checks.

        Parameters
        ----------
        ticket_id:
            The ticket identifier (e.g., 'CMH-022').
        title:
            The ticket title.
        description:
            The ticket description (from the GitHub issue body).
        phase:
            The roadmap phase (e.g., 'phase-5').
        existing_files:
            Files already recorded in the task.  Used to infer expected
            directories.

        Returns
        -------
        TicketScope
            The defined scope for this ticket.
        """
        existing_files = existing_files or []

        # Extract keywords from title and description.
        keywords = AlignmentChecker._extract_keywords(title)
        keywords |= AlignmentChecker._extract_keywords(description)

        # Extract ticket ID prefix as a keyword.
        if ticket_id:
            parts = re.split(r"[-_]", ticket_id.lower())
            for p in parts:
                if len(p) >= 2 and not p.isdigit():
                    keywords.add(p)

        # Add phase keywords.
        if phase:
            keywords |= AlignmentChecker._extract_keywords(phase)

        # Infer expected directories from existing files.
        expected_dirs: set[str] = set()
        expected_patterns: set[str] = set()
        for fp in existing_files:
            directory = _get_directory(fp)
            if directory:
                expected_dirs.add(directory)
            # Extract the filename stem as a pattern.
            basename = _get_basename(fp)
            if basename and len(basename) >= 3:
                expected_patterns.add(basename)

        # Also infer expected directories from keywords.
        # If a keyword looks like a module/package name, add common paths.
        for kw in list(keywords):
            if len(kw) >= 3 and "_" in kw:
                # snake_case keywords might be module names.
                expected_patterns.add(kw)

        return TicketScope(
            ticket_id=ticket_id,
            title=title,
            description=description,
            keywords=keywords,
            expected_directories=expected_dirs,
            expected_file_patterns=expected_patterns,
            phase=phase,
        )

    def check(
        self,
        scope: TicketScope,
        file_paths: Optional[list[str]] = None,
        actions: Optional[list[str]] = None,
    ) -> ScopeCreepReport:
        """Run scope creep detection against provided files and actions.

        This is the primary interface.  For convenience with TaskMemory,
        use :meth:`check_task`.

        Parameters
        ----------
        scope:
            The TicketScope defining what is in-scope.
        file_paths:
            List of file paths to check against the scope.
        actions:
            List of action descriptions to check against the scope.

        Returns
        -------
        ScopeCreepReport
            The scope creep assessment.
        """
        file_paths = file_paths or []
        actions = actions or []

        signals: list[ScopeCreepSignal] = []
        out_of_scope_files: list[str] = []
        out_of_scope_actions: list[str] = []

        # 1. Check file paths against scope.
        file_signals, oof = self._check_files(scope, file_paths)
        signals.extend(file_signals)
        out_of_scope_files.extend(oof)

        # 2. Check actions against scope.
        action_signals, ooa = self._check_actions(scope, actions)
        signals.extend(action_signals)
        out_of_scope_actions.extend(ooa)

        # 3. Check for new module creation outside scope.
        module_signals = self._check_new_modules(scope, file_paths)
        signals.extend(module_signals)

        # 4. Check for config file modifications.
        config_signals = self._check_config_modifications(scope, file_paths)
        signals.extend(config_signals)

        # 5. Classify severity.
        severity = self._classify_severity(signals)
        creep_detected = len(signals) > 0

        return ScopeCreepReport(
            creep_detected=creep_detected,
            severity=severity,
            out_of_scope_files=out_of_scope_files,
            out_of_scope_actions=out_of_scope_actions,
            signals=signals,
            ticket_scope_summary=scope,
            total_files_checked=len(file_paths),
            total_actions_checked=len(actions),
        )

    def check_task(
        self,
        task: "TaskMemory",
        ticket_description: str = "",
        scope_files: Optional[list[str]] = None,
    ) -> ScopeCreepReport:
        """Run scope creep detection against a TaskMemory object.

        This is a convenience wrapper that extracts file paths and actions
        from the TaskMemory, defines scope from the task metadata, and
        delegates to :meth:`check`.

        The scope is defined from the ticket metadata (title, description,
        phase) and optionally from a curated list of in-scope files.  By
        default, ``scope_files`` is empty -- the scope is inferred purely
        from keywords.  This avoids the circular problem where out-of-scope
        files in the task define themselves as in-scope.

        When ``scope_files`` is provided, those files are used to infer
        expected directories and patterns.  This is useful when the caller
        knows which files were originally planned for the ticket.

        Parameters
        ----------
        task:
            A TaskMemory instance to analyse.
        ticket_description:
            Optional ticket description (from backlog or GitHub issue).
            If empty, uses the task's metadata description.
        scope_files:
            Optional list of file paths that define the ticket's expected
            scope.  When None, scope is defined from keywords only (no
            directory inference from recorded files).

        Returns
        -------
        ScopeCreepReport
            The scope creep assessment.
        """
        description = ticket_description or task.metadata.get("description", "")

        scope = self.define_scope(
            ticket_id=task.ticket_id,
            title=task.title,
            description=description,
            phase=task.phase or "",
            existing_files=scope_files or [],
        )

        actions = [step.action for step in task.steps]
        file_paths = task.get_file_paths()

        return self.check(scope, file_paths, actions)

    def check_with_drift(
        self,
        task: "TaskMemory",
        action: str,
        file_path: Optional[str] = None,
        ticket_description: str = "",
    ) -> dict:
        """Run combined scope creep and drift detection for a single action.

        This integrates with DriftDetector to provide a unified report
        covering both temporal drift and scope creep analysis.

        Parameters
        ----------
        task:
            A TaskMemory instance providing context.
        action:
            The current action being evaluated.
        file_path:
            Optional file path the action targets.
        ticket_description:
            Optional ticket description for scope definition.

        Returns
        -------
        dict
            A dictionary with both 'scope_creep' and 'drift' reports,
            plus a combined 'overall_severity'.
        """
        from claude_code_helper_mcp.detection.drift import DriftDetector

        # Scope creep check.
        description = ticket_description or task.metadata.get("description", "")
        scope = self.define_scope(
            ticket_id=task.ticket_id,
            title=task.title,
            description=description,
            phase=task.phase or "",
            existing_files=task.get_file_paths(),
        )

        check_files = [file_path] if file_path else []
        creep_report = self.check(scope, check_files, [action])

        # Drift check.
        drift_detector = DriftDetector()
        drift_report = drift_detector.check_with_task(action, file_path, task)

        # Combined severity: take the worst of both.
        severity_order = {
            "none": 0, "low": 1, "moderate": 2, "medium": 2, "high": 3,
            "critical": 4,
        }
        creep_sev = severity_order.get(creep_report.severity, 0)
        drift_sev = severity_order.get(drift_report.severity, 0)
        max_sev = max(creep_sev, drift_sev)
        # Map back to name.
        reverse_map = {0: "none", 1: "low", 2: "medium", 3: "high", 4: "critical"}
        overall = reverse_map.get(max_sev, "none")

        return {
            "scope_creep": creep_report.to_dict(),
            "drift": drift_report.to_dict(),
            "overall_severity": overall,
            "action": action,
            "file_path": file_path,
            "ticket_id": task.ticket_id,
        }

    # ------------------------------------------------------------------
    # Detection: file scope checks
    # ------------------------------------------------------------------

    def _check_files(
        self,
        scope: TicketScope,
        file_paths: list[str],
    ) -> tuple[list[ScopeCreepSignal], list[str]]:
        """Check file paths against the ticket scope.

        Returns a tuple of (signals, out_of_scope_file_paths).
        """
        signals: list[ScopeCreepSignal] = []
        out_of_scope: list[str] = []

        for fp in file_paths:
            # Skip whitelisted files.
            if self._is_whitelisted(fp):
                continue

            # Check if the file is in an expected directory.
            in_expected_dir = self._file_in_expected_directories(fp, scope)

            # Check if the file matches expected patterns.
            matches_pattern = self._file_matches_patterns(fp, scope)

            # Check if the file path contains scope keywords.
            keyword_match = self._file_has_scope_keywords(fp, scope)

            if not in_expected_dir and not matches_pattern and not keyword_match:
                out_of_scope.append(fp)
                signals.append(ScopeCreepSignal(
                    category=CREEP_OUT_OF_SCOPE_FILE,
                    description=(
                        f"File '{fp}' is outside the expected scope for "
                        f"ticket {scope.ticket_id} ('{scope.title}'). "
                        f"It does not match any expected directory or pattern."
                    ),
                    severity_contribution=0.35,
                    evidence=(
                        f"file={fp}, "
                        f"expected_dirs={sorted(scope.expected_directories)[:5]}, "
                        f"in_expected_dir={in_expected_dir}, "
                        f"matches_pattern={matches_pattern}, "
                        f"keyword_match={keyword_match}"
                    ),
                    file_path=fp,
                ))

        return signals, out_of_scope

    # ------------------------------------------------------------------
    # Detection: action scope checks
    # ------------------------------------------------------------------

    def _check_actions(
        self,
        scope: TicketScope,
        actions: list[str],
    ) -> tuple[list[ScopeCreepSignal], list[str]]:
        """Check actions against the ticket scope keywords.

        Returns a tuple of (signals, out_of_scope_action_descriptions).
        """
        signals: list[ScopeCreepSignal] = []
        out_of_scope: list[str] = []

        if not scope.keywords:
            return signals, out_of_scope

        for action in actions:
            # Check for explicit feature addition keywords.
            if self._has_feature_addition_keywords(action):
                out_of_scope.append(action)
                signals.append(ScopeCreepSignal(
                    category=CREEP_OUT_OF_SCOPE_ACTION,
                    description=(
                        f"Action contains feature-addition language that "
                        f"suggests work beyond the ticket scope: "
                        f"'{action[:100]}'"
                    ),
                    severity_contribution=0.4,
                    evidence=f"action={action[:150]}, trigger=feature_addition_keywords",
                ))
                continue

            # Check keyword relevance.
            action_keywords = AlignmentChecker._extract_keywords(action)
            if not action_keywords:
                continue

            overlap = action_keywords & scope.keywords
            relevance = len(overlap) / len(action_keywords) if action_keywords else 0

            if relevance < self._action_threshold:
                out_of_scope.append(action)
                signals.append(ScopeCreepSignal(
                    category=CREEP_OUT_OF_SCOPE_ACTION,
                    description=(
                        f"Action has low keyword relevance to ticket scope "
                        f"({relevance:.0%} overlap, threshold is "
                        f"{self._action_threshold:.0%}): '{action[:100]}'"
                    ),
                    severity_contribution=0.25,
                    evidence=(
                        f"action_keywords={sorted(action_keywords)[:10]}, "
                        f"scope_overlap={sorted(overlap)[:10]}, "
                        f"relevance={relevance:.3f}"
                    ),
                ))

        return signals, out_of_scope

    # ------------------------------------------------------------------
    # Detection: new module creation
    # ------------------------------------------------------------------

    def _check_new_modules(
        self,
        scope: TicketScope,
        file_paths: list[str],
    ) -> list[ScopeCreepSignal]:
        """Detect creation of new modules outside the expected scope.

        A "new module" is inferred when a file is in a directory that has
        no overlap with the ticket's expected directories and the file
        creates a new Python module (not an __init__.py).
        """
        signals: list[ScopeCreepSignal] = []

        if not scope.expected_directories:
            return signals

        for fp in file_paths:
            # Skip whitelisted files.
            if self._is_whitelisted(fp):
                continue

            # Only flag Python source files as "new modules".
            if not fp.endswith(".py"):
                continue

            # Skip __init__.py files (they are package markers).
            if fp.endswith("__init__.py"):
                continue

            file_dir = _get_directory(fp)
            if not file_dir:
                continue

            # Check if this directory is outside all expected directories.
            in_scope_dir = False
            for expected_dir in scope.expected_directories:
                if file_dir.startswith(expected_dir) or expected_dir.startswith(file_dir):
                    in_scope_dir = True
                    break

            if not in_scope_dir:
                # Check if the file contains scope keywords in its path.
                if self._file_has_scope_keywords(fp, scope):
                    continue

                signals.append(ScopeCreepSignal(
                    category=CREEP_NEW_MODULE,
                    description=(
                        f"New Python module '{fp}' created in directory "
                        f"'{file_dir}' which is outside the expected scope "
                        f"directories for this ticket."
                    ),
                    severity_contribution=0.3,
                    evidence=(
                        f"file={fp}, file_dir={file_dir}, "
                        f"expected_dirs={sorted(scope.expected_directories)[:5]}"
                    ),
                    file_path=fp,
                ))

        return signals

    # ------------------------------------------------------------------
    # Detection: config file modifications
    # ------------------------------------------------------------------

    def _check_config_modifications(
        self,
        scope: TicketScope,
        file_paths: list[str],
    ) -> list[ScopeCreepSignal]:
        """Detect configuration file modifications that may be out of scope.

        Config files are only flagged when:
        1. They are not whitelisted.
        2. They are not in an expected directory for the ticket.
        3. Their path does not contain scope keywords.
        """
        signals: list[ScopeCreepSignal] = []

        for fp in file_paths:
            if not self._is_config_file(fp):
                continue

            # Config files in whitelisted patterns are always allowed.
            if self._is_whitelisted(fp):
                continue

            # Config files in expected directories are allowed.
            if self._file_in_expected_directories(fp, scope):
                continue

            # Config files whose path contains scope keywords are allowed.
            if self._file_has_scope_keywords(fp, scope):
                continue

            signals.append(ScopeCreepSignal(
                category=CREEP_CONFIG_MODIFICATION,
                description=(
                    f"Configuration file '{fp}' modified but is not "
                    f"in the expected scope for ticket {scope.ticket_id}. "
                    f"Config changes should be reviewed for relevance."
                ),
                severity_contribution=0.2,
                evidence=f"file={fp}, type=config",
                file_path=fp,
            ))

        return signals

    # ------------------------------------------------------------------
    # Severity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(signals: list[ScopeCreepSignal]) -> str:
        """Classify overall scope creep severity from detected signals.

        Uses the sum of severity contributions:
        - none: 0.0 (no signals)
        - low: < 0.3
        - medium: [0.3, 0.6)
        - high: >= 0.6
        """
        if not signals:
            return CREEP_SEVERITY_NONE

        total = sum(s.severity_contribution for s in signals)

        if total < 0.3:
            return CREEP_SEVERITY_LOW
        elif total < 0.6:
            return CREEP_SEVERITY_MEDIUM
        else:
            return CREEP_SEVERITY_HIGH

    # ------------------------------------------------------------------
    # Helpers: whitelist checking
    # ------------------------------------------------------------------

    def _is_whitelisted(self, file_path: str) -> bool:
        """Check if a file path matches any whitelist pattern.

        Returns True if the file should always be considered in-scope.
        """
        path_lower = file_path.lower()
        for pattern in self._whitelist:
            if pattern.lower() in path_lower:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers: file scope analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _file_in_expected_directories(
        file_path: str,
        scope: TicketScope,
    ) -> bool:
        """Check if a file is within any of the scope's expected directories."""
        if not scope.expected_directories:
            return False

        file_dir = _get_directory(file_path)
        if not file_dir:
            return False

        for expected_dir in scope.expected_directories:
            if file_dir.startswith(expected_dir) or expected_dir.startswith(file_dir):
                return True

        return False

    @staticmethod
    def _file_matches_patterns(
        file_path: str,
        scope: TicketScope,
    ) -> bool:
        """Check if a file matches any of the scope's expected file patterns."""
        if not scope.expected_file_patterns:
            return False

        basename = _get_basename(file_path)
        path_lower = file_path.lower()

        for pattern in scope.expected_file_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower in path_lower:
                return True
            if basename and pattern_lower in basename.lower():
                return True

        return False

    @staticmethod
    def _file_has_scope_keywords(
        file_path: str,
        scope: TicketScope,
    ) -> bool:
        """Check if a file path contains any scope keywords.

        Extracts keywords from the file path and checks for overlap
        with the scope keywords.
        """
        if not scope.keywords:
            return False

        path_keywords = AlignmentChecker._extract_path_keywords(file_path)
        if not path_keywords:
            return False

        overlap = path_keywords & scope.keywords
        return len(overlap) > 0

    @staticmethod
    def _is_config_file(file_path: str) -> bool:
        """Check if a file path points to a configuration file."""
        path_lower = file_path.lower()

        # Check by extension.
        for ext in _CONFIG_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        # Check if the file is in a config directory.
        file_dir = _get_directory(file_path)
        if file_dir:
            dir_parts = set(file_dir.lower().split("/"))
            if dir_parts & _CONFIG_DIRECTORIES:
                return True

        return False

    @staticmethod
    def _has_feature_addition_keywords(action: str) -> bool:
        """Check if an action description contains feature-addition keywords."""
        action_lower = action.lower()
        for keyword in _FEATURE_ADDITION_KEYWORDS:
            if keyword in action_lower:
                return True
        return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_directory(file_path: str) -> str:
    """Extract the directory portion of a file path (everything before the last /)."""
    idx = file_path.rfind("/")
    if idx >= 0:
        return file_path[:idx]
    return ""


def _get_basename(file_path: str) -> str:
    """Extract the filename (without extension) from a file path."""
    # Get the filename after the last /.
    idx = file_path.rfind("/")
    filename = file_path[idx + 1:] if idx >= 0 else file_path

    # Remove extension.
    dot_idx = filename.rfind(".")
    if dot_idx > 0:
        return filename[:dot_idx]
    return filename
