"""Confusion pattern detection engine for identifying agent hallucinations.

The ConfusionDetector analyses recorded steps, file references, and branch
references against the actual filesystem and recorded state to detect when
the agent exhibits confusion patterns:

- References to non-existent files (recorded as modified but not on disk)
- Wrong function or class name references (names that do not appear in target files)
- State contradictions (claiming an action was completed when evidence disagrees)

This module provides:
- Configurable detection with precision-over-recall design philosophy
- ConfusionPattern data class describing each detected pattern
- ConfusionReport with severity, patterns list, and evidence
- Integration with TaskMemory for automatic analysis of recorded data
- Filesystem verification using os.path.exists and file content scanning

Design decisions:
- All analysis is local and deterministic (no external API calls).
- ConfusionDetector is stateless per check -- call check() or check_task()
  each time.  No session state is accumulated (unlike DriftDetector).
- Precision over recall: only flag clear confusion, not ambiguous situations.
  A file that might have been deleted intentionally is not flagged.
- Filesystem checks use the project_root parameter to resolve relative paths.
- Function/class name verification scans file content with simple regex
  (no AST parsing required -- we look for def/class declarations).

Depends on:
- CMH-003: MemoryStore / file-based storage (for filesystem context)
- CMH-007: StepRecord (action, result_summary, files_involved)
- CMH-008: FileRecord (path, action), BranchRecord (branch_name, action)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Severity levels for confusion detection.
CONFUSION_SEVERITY_NONE = "none"
CONFUSION_SEVERITY_LOW = "low"
CONFUSION_SEVERITY_MEDIUM = "medium"
CONFUSION_SEVERITY_HIGH = "high"

# Pattern categories.
PATTERN_NONEXISTENT_FILE = "nonexistent_file"
PATTERN_WRONG_NAME = "wrong_name"
PATTERN_STATE_CONTRADICTION = "state_contradiction"
PATTERN_PHANTOM_STEP = "phantom_step"

# Minimum file references to trigger non-existent file detection.
# A single missing file might be intentional (e.g., a generated file).
# Two or more missing references in the same task is suspicious.
MIN_MISSING_FILES_FOR_FLAG = 1

# Regex patterns for extracting function and class names from Python files.
_PYTHON_DEF_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
    re.MULTILINE,
)
_PYTHON_CLASS_RE = re.compile(
    r"^\s*class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]",
    re.MULTILINE,
)

# Regex for extracting function/class name references from step descriptions.
_NAME_REFERENCE_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-z][a-zA-Z0-9]*)+)\b"  # PascalCase classes
    r"|\b([a-z_][a-z0-9_]*)\(\)"  # function_name() calls
)

# Keywords that indicate a "creation" claim in step descriptions.
_CREATION_KEYWORDS = frozenset({
    "created", "wrote", "generated", "added", "built", "produced",
    "initialized", "set up", "established",
})

# Keywords that indicate a "fix" or "completion" claim.
_COMPLETION_KEYWORDS = frozenset({
    "fixed", "resolved", "completed", "finished", "corrected",
    "repaired", "patched", "addressed", "handled",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConfusionPattern:
    """A single detected confusion pattern.

    Attributes:
        category: The type of confusion (e.g., 'nonexistent_file', 'wrong_name',
            'state_contradiction', 'phantom_step').
        description: Human-readable description of the confusion.
        severity_contribution: Float in [0.0, 1.0] indicating how much this
            pattern contributes to overall confusion severity.
        evidence: Specific evidence supporting the detection.
        file_path: The file involved (if applicable).
        step_number: The step number that triggered the detection (if applicable).
    """

    category: str
    description: str
    severity_contribution: float
    evidence: str = ""
    file_path: str = ""
    step_number: Optional[int] = None


@dataclass
class ConfusionReport:
    """Result of a confusion detection check.

    Attributes:
        confusion_detected: Boolean. True when at least one pattern is found.
        severity: One of 'none', 'low', 'medium', 'high'.
        patterns: List of ConfusionPattern objects describing detected issues.
        evidence: Combined evidence summary from all patterns.
        total_files_checked: Number of file references verified against filesystem.
        total_names_checked: Number of function/class names verified.
        total_steps_checked: Number of steps analysed for contradictions.
        generated_at: UTC timestamp when this report was generated.
    """

    confusion_detected: bool = False
    severity: str = CONFUSION_SEVERITY_NONE
    patterns: list[ConfusionPattern] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    total_files_checked: int = 0
    total_names_checked: int = 0
    total_steps_checked: int = 0
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "confusion_detected": self.confusion_detected,
            "severity": self.severity,
            "patterns": [
                {
                    "category": p.category,
                    "description": p.description,
                    "severity_contribution": round(p.severity_contribution, 3),
                    "evidence": p.evidence,
                    "file_path": p.file_path,
                    "step_number": p.step_number,
                }
                for p in self.patterns
            ],
            "evidence": list(self.evidence),
            "total_files_checked": self.total_files_checked,
            "total_names_checked": self.total_names_checked,
            "total_steps_checked": self.total_steps_checked,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# ConfusionDetector
# ---------------------------------------------------------------------------


class ConfusionDetector:
    """Detects confusion patterns by comparing recorded state against reality.

    The detector verifies that file references point to files that actually
    exist, that function/class names referenced in steps exist in the
    target files, and that step claims (e.g., "created file X") are
    consistent with the filesystem state.

    Parameters
    ----------
    project_root:
        Absolute path to the project root directory.  All relative file
        paths from TaskMemory are resolved against this root.
    """

    def __init__(self, project_root: str) -> None:
        if not project_root:
            raise ValueError("project_root must be a non-empty string")
        self._project_root = os.path.abspath(project_root)

    @property
    def project_root(self) -> str:
        """Return the configured project root path."""
        return self._project_root

    def check(
        self,
        file_records: Optional[list[dict]] = None,
        step_records: Optional[list[dict]] = None,
        branch_records: Optional[list[dict]] = None,
    ) -> ConfusionReport:
        """Run confusion detection against provided records.

        This is the primary low-level interface.  For convenience, use
        :meth:`check_task` with a TaskMemory object.

        Parameters
        ----------
        file_records:
            List of file record dicts with at least 'path' and 'action' keys.
        step_records:
            List of step record dicts with at least 'action', 'success',
            'files_involved', and optionally 'step_number', 'result_summary'.
        branch_records:
            List of branch record dicts with at least 'branch_name' and 'action'.

        Returns
        -------
        ConfusionReport
            The confusion assessment.
        """
        file_records = file_records or []
        step_records = step_records or []
        branch_records = branch_records or []

        patterns: list[ConfusionPattern] = []
        files_checked = 0
        names_checked = 0
        steps_checked = len(step_records)

        # 1. Check for non-existent file references.
        file_patterns, fc = self._check_file_references(file_records, step_records)
        patterns.extend(file_patterns)
        files_checked += fc

        # 2. Check for wrong function/class name references.
        name_patterns, nc = self._check_name_references(step_records)
        patterns.extend(name_patterns)
        names_checked += nc

        # 3. Check for state contradictions.
        contradiction_patterns = self._check_state_contradictions(
            file_records, step_records
        )
        patterns.extend(contradiction_patterns)

        # 4. Check for phantom steps (steps referencing non-existent prior steps).
        phantom_patterns = self._check_phantom_references(step_records)
        patterns.extend(phantom_patterns)

        # 5. Build the report.
        severity = self._classify_severity(patterns)
        confusion_detected = len(patterns) > 0
        evidence_list = [p.evidence for p in patterns if p.evidence]

        return ConfusionReport(
            confusion_detected=confusion_detected,
            severity=severity,
            patterns=patterns,
            evidence=evidence_list,
            total_files_checked=files_checked,
            total_names_checked=names_checked,
            total_steps_checked=steps_checked,
        )

    def check_task(self, task: "TaskMemory") -> ConfusionReport:
        """Run confusion detection against a TaskMemory object.

        This is a convenience wrapper that extracts records from the
        TaskMemory and delegates to :meth:`check`.

        Parameters
        ----------
        task:
            A TaskMemory instance to analyse.

        Returns
        -------
        ConfusionReport
            The confusion assessment.
        """
        file_records = [
            {"path": f.path, "action": f.action.value}
            for f in task.files
        ]
        step_records = [
            {
                "action": s.action,
                "success": s.success,
                "files_involved": list(s.files_involved),
                "step_number": s.step_number,
                "result_summary": s.result_summary or "",
                "tool_used": s.tool_used or "",
                "description": s.description,
            }
            for s in task.steps
        ]
        branch_records = [
            {"branch_name": b.branch_name, "action": b.action.value}
            for b in task.branches
        ]
        return self.check(file_records, step_records, branch_records)

    # ------------------------------------------------------------------
    # Detection: non-existent file references
    # ------------------------------------------------------------------

    def _check_file_references(
        self,
        file_records: list[dict],
        step_records: list[dict],
    ) -> tuple[list[ConfusionPattern], int]:
        """Check that referenced files actually exist on disk.

        Scans both file_records (explicit file tracking) and step_records
        (files_involved lists).  Only flags files that were recorded as
        created or modified but do not exist.  Files recorded as deleted
        are expected to be missing.

        Returns a tuple of (patterns, files_checked_count).
        """
        patterns: list[ConfusionPattern] = []
        files_checked = 0

        # Collect all unique file paths and their expected state.
        # Key: relative path, Value: last known action
        file_state: dict[str, str] = {}

        for rec in file_records:
            path = rec.get("path", "")
            action = rec.get("action", "")
            if path:
                file_state[path] = action

        # Also collect files from step records.
        step_files: set[str] = set()
        for step in step_records:
            for fp in step.get("files_involved", []):
                if fp:
                    step_files.add(fp)

        # Check files from file_records first (these have explicit action info).
        for path, action in file_state.items():
            abs_path = self._resolve_path(path)
            files_checked += 1

            # Skip files that were deleted -- they are expected to be missing.
            if action in ("deleted",):
                continue

            # Skip files that were only read -- they may have been transient.
            if action in ("read",):
                # Still check if it exists, but with lower severity.
                if not os.path.exists(abs_path):
                    patterns.append(ConfusionPattern(
                        category=PATTERN_NONEXISTENT_FILE,
                        description=(
                            f"File '{path}' was recorded as read but does not "
                            f"exist on disk. It may have been moved or deleted "
                            f"outside of task tracking."
                        ),
                        severity_contribution=0.15,
                        evidence=f"action=read, resolved_path={abs_path}",
                        file_path=path,
                    ))
                continue

            # For created/modified/renamed files, they should exist.
            if not os.path.exists(abs_path):
                patterns.append(ConfusionPattern(
                    category=PATTERN_NONEXISTENT_FILE,
                    description=(
                        f"File '{path}' was recorded as {action} but does "
                        f"not exist on disk. The agent may have referenced "
                        f"a wrong path or the file was not actually created."
                    ),
                    severity_contribution=0.4,
                    evidence=(
                        f"action={action}, resolved_path={abs_path}, "
                        f"exists=False"
                    ),
                    file_path=path,
                ))

        # Check files from steps that are not in file_records.
        for path in step_files:
            if path in file_state:
                continue  # Already checked above.
            abs_path = self._resolve_path(path)
            files_checked += 1

            if not os.path.exists(abs_path):
                patterns.append(ConfusionPattern(
                    category=PATTERN_NONEXISTENT_FILE,
                    description=(
                        f"File '{path}' is referenced in step files_involved "
                        f"but does not exist on disk."
                    ),
                    severity_contribution=0.25,
                    evidence=f"source=step_files_involved, resolved_path={abs_path}",
                    file_path=path,
                ))

        return patterns, files_checked

    # ------------------------------------------------------------------
    # Detection: wrong function/class name references
    # ------------------------------------------------------------------

    def _check_name_references(
        self,
        step_records: list[dict],
    ) -> tuple[list[ConfusionPattern], int]:
        """Check that function/class names referenced in steps exist in files.

        Scans step action descriptions and result summaries for PascalCase
        class names and function_name() call patterns.  For each name found,
        checks whether it appears in any of the files involved in that step.

        This is a precision-focused check: it only flags names when:
        1. The step explicitly references specific files (files_involved)
        2. Those files exist and are readable
        3. The referenced name does not appear in those files

        Returns a tuple of (patterns, names_checked_count).
        """
        patterns: list[ConfusionPattern] = []
        names_checked = 0

        for step in step_records:
            action = step.get("action", "")
            result = step.get("result_summary", "")
            description = step.get("description", "")
            files_involved = step.get("files_involved", [])
            step_number = step.get("step_number")

            if not files_involved:
                continue

            # Extract name references from action + result + description.
            text = f"{action} {result} {description}"
            referenced_names = self._extract_name_references(text)

            if not referenced_names:
                continue

            # Build a set of names defined in the involved files.
            defined_names: set[str] = set()
            readable_files: list[str] = []

            for fp in files_involved:
                abs_path = self._resolve_path(fp)
                if not os.path.exists(abs_path):
                    continue
                if not abs_path.endswith(".py"):
                    continue  # Only scan Python files for now.
                try:
                    content = self._read_file_content(abs_path)
                    defined_names |= self._extract_defined_names(content)
                    readable_files.append(fp)
                except (OSError, UnicodeDecodeError):
                    continue

            if not readable_files:
                continue

            # Check each referenced name against defined names.
            for name in referenced_names:
                names_checked += 1
                # Skip very common names that are likely not specific references.
                if name.lower() in _COMMON_NAMES:
                    continue
                if name not in defined_names:
                    # Do a fuzzy check -- the name might be close to a defined name.
                    close_match = self._find_close_match(name, defined_names)
                    if close_match:
                        patterns.append(ConfusionPattern(
                            category=PATTERN_WRONG_NAME,
                            description=(
                                f"Step references '{name}' but this name does "
                                f"not appear in the involved files. Did you mean "
                                f"'{close_match}'?"
                            ),
                            severity_contribution=0.3,
                            evidence=(
                                f"referenced_name={name}, "
                                f"closest_match={close_match}, "
                                f"files_checked={readable_files}"
                            ),
                            file_path=readable_files[0] if readable_files else "",
                            step_number=step_number,
                        ))
                    else:
                        patterns.append(ConfusionPattern(
                            category=PATTERN_WRONG_NAME,
                            description=(
                                f"Step references '{name}' but this name is not "
                                f"defined in any of the involved files: "
                                f"{readable_files}."
                            ),
                            severity_contribution=0.35,
                            evidence=(
                                f"referenced_name={name}, "
                                f"defined_names_sample="
                                f"{sorted(defined_names)[:10]}, "
                                f"files_checked={readable_files}"
                            ),
                            file_path=readable_files[0] if readable_files else "",
                            step_number=step_number,
                        ))

        return patterns, names_checked

    # ------------------------------------------------------------------
    # Detection: state contradictions
    # ------------------------------------------------------------------

    def _check_state_contradictions(
        self,
        file_records: list[dict],
        step_records: list[dict],
    ) -> list[ConfusionPattern]:
        """Detect contradictions between step claims and filesystem state.

        Checks for:
        - "Created file X" but file X does not exist
        - "Fixed the bug" but subsequent steps show the same error
        - Steps claiming success on non-existent targets
        """
        patterns: list[ConfusionPattern] = []

        for step in step_records:
            action_lower = step.get("action", "").lower()
            result_lower = (step.get("result_summary", "") or "").lower()
            files_involved = step.get("files_involved", [])
            success = step.get("success", True)
            step_number = step.get("step_number")

            # Check: "created file X" but X does not exist.
            if success and any(kw in action_lower for kw in _CREATION_KEYWORDS):
                for fp in files_involved:
                    abs_path = self._resolve_path(fp)
                    if not os.path.exists(abs_path):
                        patterns.append(ConfusionPattern(
                            category=PATTERN_STATE_CONTRADICTION,
                            description=(
                                f"Step claims to have created/written '{fp}' "
                                f"(marked as success) but the file does not exist."
                            ),
                            severity_contribution=0.5,
                            evidence=(
                                f"action={step.get('action', '')}, "
                                f"success=True, file_exists=False, "
                                f"resolved_path={abs_path}"
                            ),
                            file_path=fp,
                            step_number=step_number,
                        ))

        # Check for "fixed" claims followed by similar failures.
        fix_claims = []
        for i, step in enumerate(step_records):
            action_lower = step.get("action", "").lower()
            if step.get("success", True) and any(
                kw in action_lower for kw in _COMPLETION_KEYWORDS
            ):
                fix_claims.append(i)

        for fix_idx in fix_claims:
            fix_step = step_records[fix_idx]
            fix_action = fix_step.get("action", "").lower()
            fix_files = set(fix_step.get("files_involved", []))

            # Look at subsequent steps for failures on the same files.
            for j in range(fix_idx + 1, min(fix_idx + 4, len(step_records))):
                subsequent = step_records[j]
                if subsequent.get("success", True):
                    continue
                sub_files = set(subsequent.get("files_involved", []))
                overlap = fix_files & sub_files
                if overlap:
                    patterns.append(ConfusionPattern(
                        category=PATTERN_STATE_CONTRADICTION,
                        description=(
                            f"Step {fix_step.get('step_number', '?')} claims "
                            f"'{fix_step.get('action', '')}' (success), but "
                            f"step {subsequent.get('step_number', '?')} failed "
                            f"on overlapping files: {sorted(overlap)}."
                        ),
                        severity_contribution=0.35,
                        evidence=(
                            f"fix_step={fix_step.get('step_number')}, "
                            f"failure_step={subsequent.get('step_number')}, "
                            f"overlapping_files={sorted(overlap)}"
                        ),
                        file_path=sorted(overlap)[0] if overlap else "",
                        step_number=subsequent.get("step_number"),
                    ))
                    break  # One contradiction per fix claim is enough.

        return patterns

    # ------------------------------------------------------------------
    # Detection: phantom step references
    # ------------------------------------------------------------------

    def _check_phantom_references(
        self,
        step_records: list[dict],
    ) -> list[ConfusionPattern]:
        """Detect steps that reference completing a prior step that never happened.

        Looks for result summaries or descriptions that reference specific
        step numbers or actions that are not in the step history.
        """
        patterns: list[ConfusionPattern] = []

        # Build index of recorded step numbers and action keywords.
        recorded_step_numbers = {
            s.get("step_number") for s in step_records if s.get("step_number")
        }

        for step in step_records:
            result = (step.get("result_summary", "") or "")
            description = step.get("description", "")
            text = f"{result} {description}"

            # Look for "step N" or "step #N" references.
            step_refs = re.findall(r"step\s+#?(\d+)", text, re.IGNORECASE)
            for ref_str in step_refs:
                ref_num = int(ref_str)
                step_number = step.get("step_number")
                if (
                    ref_num not in recorded_step_numbers
                    and ref_num > 0
                    and (step_number is None or ref_num != step_number)
                ):
                    patterns.append(ConfusionPattern(
                        category=PATTERN_PHANTOM_STEP,
                        description=(
                            f"Step {step_number} references 'step {ref_num}' "
                            f"which does not exist in the recorded step history."
                        ),
                        severity_contribution=0.25,
                        evidence=(
                            f"referencing_step={step_number}, "
                            f"phantom_step_ref={ref_num}, "
                            f"recorded_steps={sorted(recorded_step_numbers)}"
                        ),
                        step_number=step_number,
                    ))

        return patterns

    # ------------------------------------------------------------------
    # Severity classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_severity(patterns: list[ConfusionPattern]) -> str:
        """Classify overall confusion severity from detected patterns.

        Uses the sum of severity contributions to determine the level:
        - none: 0.0 (no patterns)
        - low: < 0.3
        - medium: [0.3, 0.6)
        - high: >= 0.6
        """
        if not patterns:
            return CONFUSION_SEVERITY_NONE

        total = sum(p.severity_contribution for p in patterns)

        if total < 0.3:
            return CONFUSION_SEVERITY_LOW
        elif total < 0.6:
            return CONFUSION_SEVERITY_MEDIUM
        else:
            return CONFUSION_SEVERITY_HIGH

    # ------------------------------------------------------------------
    # Helpers: path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self, relative_path: str) -> str:
        """Resolve a relative file path against the project root.

        If the path is already absolute, returns it as-is.  Otherwise,
        joins it with the project root.
        """
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self._project_root, relative_path)

    # ------------------------------------------------------------------
    # Helpers: name extraction and matching
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_name_references(text: str) -> set[str]:
        """Extract function/class name references from text.

        Looks for:
        - PascalCase names (likely class references): ConfusionDetector, TaskMemory
        - function_call() patterns: check_alignment(), record_step()
        """
        names: set[str] = set()

        # PascalCase class names (at least two capital letters).
        for match in re.finditer(
            r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-z][a-zA-Z0-9]*)+)\b", text
        ):
            name = match.group(1)
            # Filter out common non-class patterns.
            if not _is_likely_class_name(name):
                continue
            names.add(name)

        # function_name() patterns.
        for match in re.finditer(r"\b([a-z_][a-z0-9_]{2,})\(\)", text):
            names.add(match.group(1))

        return names

    @staticmethod
    def _extract_defined_names(file_content: str) -> set[str]:
        """Extract defined function and class names from Python file content.

        Uses regex to find `def name(` and `class Name(` / `class Name:`
        declarations.  This is intentionally simple (no AST) for robustness
        against syntax errors in incomplete files.
        """
        names: set[str] = set()

        for match in _PYTHON_DEF_RE.finditer(file_content):
            names.add(match.group(1))

        for match in _PYTHON_CLASS_RE.finditer(file_content):
            names.add(match.group(1))

        return names

    @staticmethod
    def _read_file_content(abs_path: str, max_bytes: int = 512_000) -> str:
        """Read file content up to max_bytes.

        Reads only the first max_bytes to prevent memory issues with very
        large files.  For name checking, the first 500KB is more than enough.
        """
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)

    @staticmethod
    def _find_close_match(
        name: str,
        candidates: set[str],
        max_distance: int = 2,
    ) -> Optional[str]:
        """Find the closest matching name from a set of candidates.

        Uses a simple character-level comparison.  Returns the closest
        match if the edit distance is within max_distance, else None.
        """
        if not candidates:
            return None

        best_match: Optional[str] = None
        best_distance = max_distance + 1

        name_lower = name.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()
            dist = _levenshtein_distance(name_lower, candidate_lower)
            if dist < best_distance:
                best_distance = dist
                best_match = candidate

        if best_distance <= max_distance:
            return best_match
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Common names that should be ignored in name reference checks.
_COMMON_NAMES: frozenset[str] = frozenset({
    "self", "cls", "none", "true", "false", "str", "int", "float", "bool",
    "list", "dict", "set", "tuple", "type", "object", "exception",
    "valueerror", "typeerror", "keyerror", "indexerror", "oserror",
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "isinstance", "issubclass", "super", "property", "staticmethod",
    "classmethod", "abstractmethod", "dataclass", "field",
    "optional", "union", "any",
})


def _is_likely_class_name(name: str) -> bool:
    """Determine if a PascalCase string is likely a class name reference.

    Filters out common English words that happen to match PascalCase
    (e.g., "FileRecord" is likely a class, "ThePython" is not).
    """
    # Must have at least 4 characters.
    if len(name) < 4:
        return False

    # Must contain at least one lowercase letter after the first character.
    if not any(c.islower() for c in name[1:]):
        return False

    # Filter out common non-class PascalCase words.
    excluded = {
        "TypeError", "ValueError", "KeyError", "IndexError",
        "OSError", "IOError", "RuntimeError", "ImportError",
        "AttributeError", "NotImplementedError", "StopIteration",
        "BaseException", "Exception",
    }
    if name in excluded:
        return False

    return True


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses the classic dynamic programming approach.  For the short strings
    we deal with (identifiers), this is efficient enough.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
