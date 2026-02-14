"""Alignment checking engine for comparing actions against task scope.

The AlignmentChecker evaluates whether a proposed or current action is consistent
with the active task's defined scope. It uses keyword extraction, file path
analysis, and contextual scoring to produce an AlignmentReport with a confidence
score, warnings, and scope metadata.

This module provides the core logic used by the check_alignment MCP tool
(CMH-012). The MCP tool is a thin wrapper that calls AlignmentChecker and
formats the result.

Design decisions:
- All analysis is local and deterministic (no external API calls).
- Keyword extraction uses simple tokenisation and normalisation (no NLP deps).
- Confidence score is a float in [0.0, 1.0] where 1.0 means perfectly aligned.
- Warnings are human-readable strings describing potential scope drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    """Result of an alignment check.

    Attributes:
        confidence: Float in [0.0, 1.0]. 1.0 = perfectly aligned with task scope.
        aligned: Boolean shortcut. True when confidence >= threshold (default 0.5).
        warnings: Human-readable warnings about potential scope drift.
        scope_info: Metadata about the task scope used for comparison.
        action_analysis: Details about how the action was analysed.
        generated_at: UTC timestamp when this report was generated.
    """

    confidence: float
    aligned: bool
    warnings: list[str] = field(default_factory=list)
    scope_info: dict = field(default_factory=dict)
    action_analysis: dict = field(default_factory=dict)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "confidence": round(self.confidence, 3),
            "aligned": self.aligned,
            "warnings": list(self.warnings),
            "scope_info": dict(self.scope_info),
            "action_analysis": dict(self.action_analysis),
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default alignment threshold. Actions below this score are flagged.
DEFAULT_ALIGNMENT_THRESHOLD = 0.5

# Words stripped from keyword extraction (common English stop words + code noise).
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "i", "we", "they", "you", "he", "she",
    "my", "our", "your", "their", "not", "no", "so", "if", "then", "else",
    "when", "while", "as", "up", "out", "about", "into", "over", "after",
    "before", "between", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "also", "just", "than",
    "very", "too", "now", "new", "use", "using", "used", "add", "added",
    "adding", "create", "created", "creating", "implement", "implemented",
    "implementing", "update", "updated", "updating", "file", "files",
    "code", "module", "class", "function", "method", "test", "tests",
})


# ---------------------------------------------------------------------------
# AlignmentChecker
# ---------------------------------------------------------------------------


class AlignmentChecker:
    """Evaluates alignment between an action and a task's scope.

    The checker extracts keywords from the task's title, description, phase,
    recorded files, and recorded steps, then compares them against keywords
    extracted from the action being checked. The overlap ratio, combined with
    file-path relevance and contextual signals, produces a confidence score.

    Parameters
    ----------
    threshold:
        The minimum confidence score for an action to be considered aligned.
        Defaults to 0.5.
    """

    def __init__(self, threshold: float = DEFAULT_ALIGNMENT_THRESHOLD) -> None:
        self.threshold = max(0.0, min(1.0, threshold))

    def check(
        self,
        action: str,
        file_path: Optional[str],
        task_title: str,
        task_description: str = "",
        task_phase: Optional[str] = None,
        task_files: Optional[list[str]] = None,
        task_steps: Optional[list[str]] = None,
        task_ticket_id: Optional[str] = None,
    ) -> AlignmentReport:
        """Run an alignment check.

        Parameters
        ----------
        action:
            Description of the action being checked (e.g., "Adding error
            handling to the REST client").
        file_path:
            Optional file path the action targets. Used for file-scope analysis.
        task_title:
            The active task's title.
        task_description:
            Optional task description (from metadata or ticket body).
        task_phase:
            Optional roadmap phase (e.g., "phase-3").
        task_files:
            List of file paths already recorded in the task.
        task_steps:
            List of action strings from steps already recorded.
        task_ticket_id:
            The ticket identifier (e.g., "CMH-012").

        Returns
        -------
        AlignmentReport
            The alignment assessment with confidence score and warnings.
        """
        task_files = task_files or []
        task_steps = task_steps or []

        # 1. Extract scope keywords from the task context.
        scope_keywords = self._extract_scope_keywords(
            title=task_title,
            description=task_description,
            phase=task_phase,
            file_paths=task_files,
            step_actions=task_steps,
            ticket_id=task_ticket_id,
        )

        # 2. Extract action keywords.
        action_keywords = self._extract_keywords(action)
        if file_path:
            action_keywords |= self._extract_path_keywords(file_path)

        # 3. Compute component scores.
        keyword_score = self._keyword_overlap_score(action_keywords, scope_keywords)
        file_score = self._file_relevance_score(file_path, task_files)
        context_score = self._contextual_score(action, task_title, task_ticket_id)

        # 4. Weighted combination.
        #    keyword_overlap: 40%, file_relevance: 30%, context: 30%
        confidence = (
            0.40 * keyword_score
            + 0.30 * file_score
            + 0.30 * context_score
        )
        confidence = max(0.0, min(1.0, confidence))

        # 5. Generate warnings.
        warnings = self._generate_warnings(
            confidence=confidence,
            action=action,
            file_path=file_path,
            task_title=task_title,
            task_files=task_files,
            keyword_score=keyword_score,
            file_score=file_score,
            context_score=context_score,
        )

        # 6. Build scope_info and action_analysis for transparency.
        scope_info = {
            "ticket_id": task_ticket_id,
            "title": task_title,
            "phase": task_phase,
            "files_in_scope": task_files[:20],  # cap for readability
            "scope_keyword_count": len(scope_keywords),
            "scope_keywords_sample": sorted(scope_keywords)[:15],
        }

        action_analysis = {
            "action": action,
            "file_path": file_path,
            "action_keyword_count": len(action_keywords),
            "action_keywords_sample": sorted(action_keywords)[:15],
            "keyword_overlap_score": round(keyword_score, 3),
            "file_relevance_score": round(file_score, 3),
            "contextual_score": round(context_score, 3),
            "threshold": self.threshold,
        }

        aligned = confidence >= self.threshold

        return AlignmentReport(
            confidence=confidence,
            aligned=aligned,
            warnings=warnings,
            scope_info=scope_info,
            action_analysis=action_analysis,
        )

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract meaningful keywords from a text string.

        Tokenises on non-alphanumeric boundaries, lowercases, strips stop
        words, and discards very short tokens (< 2 chars) and pure numbers.
        """
        if not text:
            return set()

        # Split on non-alphanumeric (including underscores and hyphens as delimiters).
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        keywords = set()
        for token in tokens:
            token = token.strip()
            if len(token) < 2:
                continue
            if token.isdigit():
                continue
            if token in _STOP_WORDS:
                continue
            keywords.add(token)

        # Also extract camelCase / PascalCase sub-words.
        camel_tokens = re.findall(r"[A-Z][a-z]+|[a-z]+", text)
        for ct in camel_tokens:
            ct_lower = ct.lower()
            if len(ct_lower) >= 2 and ct_lower not in _STOP_WORDS and not ct_lower.isdigit():
                keywords.add(ct_lower)

        return keywords

    @staticmethod
    def _extract_path_keywords(file_path: str) -> set[str]:
        """Extract keywords from a file path.

        Splits on path separators, dots, underscores, and hyphens.
        Discards common non-informative segments like 'src', 'tests', 'py'.
        """
        if not file_path:
            return set()

        non_informative = {
            "src", "tests", "test", "py", "js", "ts", "json", "md", "yaml",
            "yml", "cfg", "ini", "txt", "toml", "lock", "css", "html",
            "__init__", "init", "__pycache__",
        }

        segments = re.split(r"[/\\._\-]+", file_path.lower())
        keywords = set()
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 2:
                continue
            if seg in non_informative:
                continue
            if seg.isdigit():
                continue
            keywords.add(seg)

        return keywords

    def _extract_scope_keywords(
        self,
        title: str,
        description: str,
        phase: Optional[str],
        file_paths: list[str],
        step_actions: list[str],
        ticket_id: Optional[str],
    ) -> set[str]:
        """Build the full set of scope keywords from all task context."""
        keywords = set()

        # From title (highest signal).
        keywords |= self._extract_keywords(title)

        # From description.
        keywords |= self._extract_keywords(description)

        # From phase.
        if phase:
            keywords |= self._extract_keywords(phase)

        # From recorded file paths.
        for fp in file_paths:
            keywords |= self._extract_path_keywords(fp)

        # From recorded step actions (limited to prevent noise from old steps).
        for action_text in step_actions[-10:]:
            keywords |= self._extract_keywords(action_text)

        # From ticket ID (extract the project prefix and number).
        if ticket_id:
            parts = re.split(r"[-_]", ticket_id.lower())
            for p in parts:
                if len(p) >= 2 and not p.isdigit():
                    keywords.add(p)

        return keywords

    # ------------------------------------------------------------------
    # Scoring functions
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_overlap_score(
        action_keywords: set[str],
        scope_keywords: set[str],
    ) -> float:
        """Compute the Jaccard-like overlap score between action and scope keywords.

        Uses the ratio of intersection to action keywords (not union), because
        we want to know what fraction of the action's intent matches the scope.
        A task with many scope keywords should not penalise a focused action.
        """
        if not action_keywords:
            # No action keywords means we cannot assess -- return neutral.
            return 0.5

        if not scope_keywords:
            # No scope keywords means no context to compare against -- neutral.
            return 0.5

        overlap = action_keywords & scope_keywords
        # Weight by ratio of action keywords that appear in scope.
        coverage = len(overlap) / len(action_keywords)
        return min(1.0, coverage)

    @staticmethod
    def _file_relevance_score(
        file_path: Optional[str],
        task_files: list[str],
    ) -> float:
        """Score how relevant a file path is to the task's recorded files.

        Returns 1.0 if the file is already recorded in the task, a partial
        score if it shares directory prefixes with recorded files, and 0.5
        (neutral) if no file path is provided.
        """
        if not file_path:
            return 0.5  # neutral when no file is specified

        if not task_files:
            # No files recorded yet -- first file gets a neutral-positive score.
            return 0.6

        # Exact match -- this file is already in the task.
        if file_path in task_files:
            return 1.0

        # Check directory overlap: how many recorded files share a common prefix.
        file_dir = _get_directory(file_path)
        matching_dirs = 0
        for tf in task_files:
            tf_dir = _get_directory(tf)
            if file_dir and tf_dir and (
                file_dir.startswith(tf_dir) or tf_dir.startswith(file_dir)
            ):
                matching_dirs += 1

        if matching_dirs > 0:
            # Partial match based on directory overlap.
            return min(1.0, 0.6 + 0.1 * matching_dirs)

        # No directory overlap -- could be out of scope.
        return 0.2

    @staticmethod
    def _contextual_score(
        action: str,
        task_title: str,
        task_ticket_id: Optional[str],
    ) -> float:
        """Compute a contextual score based on direct references.

        Checks if the action explicitly mentions the task ticket ID or
        contains strong contextual signals that tie it to the task.
        """
        action_lower = action.lower()
        title_lower = task_title.lower()
        score = 0.5  # start neutral

        # Direct ticket ID reference is a strong signal.
        if task_ticket_id and task_ticket_id.lower() in action_lower:
            score += 0.3

        # Check if key title words appear in the action.
        title_keywords = set(re.split(r"[^a-zA-Z0-9]+", title_lower))
        title_keywords -= _STOP_WORDS
        title_keywords = {w for w in title_keywords if len(w) >= 3}

        if title_keywords:
            action_words = set(re.split(r"[^a-zA-Z0-9]+", action_lower))
            title_overlap = title_keywords & action_words
            if title_overlap:
                score += 0.2 * (len(title_overlap) / len(title_keywords))

        return min(1.0, score)

    # ------------------------------------------------------------------
    # Warning generation
    # ------------------------------------------------------------------

    def _generate_warnings(
        self,
        confidence: float,
        action: str,
        file_path: Optional[str],
        task_title: str,
        task_files: list[str],
        keyword_score: float,
        file_score: float,
        context_score: float,
    ) -> list[str]:
        """Generate human-readable warnings based on score analysis."""
        warnings: list[str] = []

        # Low overall confidence.
        if confidence < self.threshold:
            warnings.append(
                f"Action may be outside task scope (confidence: {confidence:.1%}). "
                f"Task: '{task_title}'."
            )

        # File outside recorded scope.
        if file_path and file_score < 0.4 and task_files:
            recorded_dirs = sorted({_get_directory(f) for f in task_files if _get_directory(f)})
            warnings.append(
                f"File '{file_path}' is outside the directories previously "
                f"touched by this task. Recorded directories: {recorded_dirs[:5]}."
            )

        # No keyword overlap.
        if keyword_score < 0.2:
            warnings.append(
                "Very low keyword overlap between action and task scope. "
                "The action description may be unrelated to the current task."
            )

        # Moderate confidence -- advisory.
        if self.threshold <= confidence < 0.7 and not warnings:
            warnings.append(
                f"Action is marginally aligned (confidence: {confidence:.1%}). "
                f"Consider verifying this action is within scope."
            )

        return warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_directory(file_path: str) -> str:
    """Extract the directory portion of a file path (everything before the last /)."""
    idx = file_path.rfind("/")
    if idx >= 0:
        return file_path[:idx]
    return ""
