"""Drift, error loop, confusion, and scope creep detection engines.

This package provides detection engines for identifying when agent actions
diverge from the active task's scope:

- :class:`AlignmentChecker` -- single-action alignment checking (CMH-012)
- :class:`DriftDetector` -- multi-action drift detection with temporal tracking (CMH-019)
"""

from claude_code_helper_mcp.detection.alignment import (
    AlignmentChecker,
    AlignmentReport,
    DEFAULT_ALIGNMENT_THRESHOLD,
)
from claude_code_helper_mcp.detection.drift import (
    DEFAULT_THRESHOLDS,
    DriftDetector,
    DriftIndicator,
    DriftReport,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_NONE,
)

__all__ = [
    "AlignmentChecker",
    "AlignmentReport",
    "DEFAULT_ALIGNMENT_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "DriftDetector",
    "DriftIndicator",
    "DriftReport",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MODERATE",
    "SEVERITY_NONE",
]
