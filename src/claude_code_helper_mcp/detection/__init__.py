"""Drift, error loop, confusion, scope creep detection engines, and intervention system.

This package provides detection engines for identifying when agent actions
diverge from the active task's scope:

- :class:`AlignmentChecker` -- single-action alignment checking (CMH-012)
- :class:`DriftDetector` -- multi-action drift detection with temporal tracking (CMH-019)
- :class:`ErrorLoopDetector` -- consecutive failure loop detection (CMH-020)
- :class:`ConfusionDetector` -- confusion pattern detection against filesystem (CMH-021)
- :class:`ScopeCreepDetector` -- scope creep detection against ticket scope (CMH-022)
- :class:`InterventionManager` -- graduated intervention system aggregating all detectors (CMH-023)
"""

from claude_code_helper_mcp.detection.alignment import (
    AlignmentChecker,
    AlignmentReport,
    DEFAULT_ALIGNMENT_THRESHOLD,
)
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
from claude_code_helper_mcp.detection.error_loop import (
    DEFAULT_LOOP_THRESHOLD,
    ErrorLoopDetector,
    ErrorLoopReport,
    FailureRecord,
    LoopEvidence,
    LOOP_SEVERITY_ACTIVE,
    LOOP_SEVERITY_CRITICAL,
    LOOP_SEVERITY_NONE,
    LOOP_SEVERITY_WARNING,
    SIMILARITY_THRESHOLD,
)
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
)
from claude_code_helper_mcp.detection.intervention import (
    INTERVENTION_LEVEL_ESCALATION,
    INTERVENTION_LEVEL_NONE,
    INTERVENTION_LEVEL_WARNING,
    AggregatedDetectionReport,
    DetectionSummary,
    InterventionManager,
    InterventionResponse,
)

__all__ = [
    "AggregatedDetectionReport",
    "AlignmentChecker",
    "AlignmentReport",
    "CONFUSION_SEVERITY_HIGH",
    "CONFUSION_SEVERITY_LOW",
    "CONFUSION_SEVERITY_MEDIUM",
    "CONFUSION_SEVERITY_NONE",
    "ConfusionDetector",
    "ConfusionPattern",
    "ConfusionReport",
    "DEFAULT_ALIGNMENT_THRESHOLD",
    "DEFAULT_LOOP_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "DetectionSummary",
    "DriftDetector",
    "DriftIndicator",
    "DriftReport",
    "ErrorLoopDetector",
    "ErrorLoopReport",
    "FailureRecord",
    "INTERVENTION_LEVEL_ESCALATION",
    "INTERVENTION_LEVEL_NONE",
    "INTERVENTION_LEVEL_WARNING",
    "InterventionManager",
    "InterventionResponse",
    "LoopEvidence",
    "LOOP_SEVERITY_ACTIVE",
    "LOOP_SEVERITY_CRITICAL",
    "LOOP_SEVERITY_NONE",
    "LOOP_SEVERITY_WARNING",
    "PATTERN_NONEXISTENT_FILE",
    "PATTERN_PHANTOM_STEP",
    "PATTERN_STATE_CONTRADICTION",
    "PATTERN_WRONG_NAME",
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MODERATE",
    "SEVERITY_NONE",
    "SIMILARITY_THRESHOLD",
    "CREEP_CONFIG_MODIFICATION",
    "CREEP_NEW_MODULE",
    "CREEP_OUT_OF_SCOPE_ACTION",
    "CREEP_OUT_OF_SCOPE_FILE",
    "CREEP_SEVERITY_HIGH",
    "CREEP_SEVERITY_LOW",
    "CREEP_SEVERITY_MEDIUM",
    "CREEP_SEVERITY_NONE",
    "ScopeCreepDetector",
    "ScopeCreepReport",
    "ScopeCreepSignal",
    "TicketScope",
]
