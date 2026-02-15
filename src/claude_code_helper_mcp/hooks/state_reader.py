"""Pipeline state file readers for Agent Forge integration.

Provides the :class:`StateReader` class which reads Agent Forge state files
(pipeline.json, build-output.json, backlog.json, plan-output.json) to obtain
current pipeline context.  This enables the memory system to understand what
the pipeline is doing without requiring explicit agent communication.

Key design decisions:

- **Caching**: State files are read once per task and cached in-memory.
  The cache can be invalidated by calling :meth:`StateReader.invalidate_cache`
  or by calling :meth:`StateReader.refresh`.
- **Graceful degradation**: Missing or malformed files return *None* or
  typed default objects.  No exceptions propagate to callers.
- **Path resolution**: The reader locates ``.agent-forge/`` from a
  configurable project root, with fallback to auto-detection.

Typical usage::

    reader = StateReader("/path/to/project")

    # Individual readers
    pipeline = reader.read_pipeline()
    build = reader.read_build_output()
    backlog = reader.read_backlog()
    plan = reader.read_plan_output()

    # Convenience aggregator
    context = reader.get_current_ticket_context()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The Agent Forge config directory name, expected at the project root.
AGENT_FORGE_DIR = ".agent-forge"

# Relative paths to state files within .agent-forge/
PIPELINE_STATE_PATH = "state/pipeline.json"
BUILD_OUTPUT_PATH = "state/build-output.json"
PLAN_OUTPUT_PATH = "state/plan-output.json"
BACKLOG_PATH = "plans/backlog.json"

# Default cache TTL in seconds.  State files are re-read from disk only
# after this interval has elapsed since the last read.
DEFAULT_CACHE_TTL_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Data classes for typed state file contents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineState:
    """Typed representation of key fields from pipeline.json.

    Only the fields that are relevant to the memory system are extracted.
    The full JSON is available via :attr:`raw` for advanced consumers.
    """

    status: str
    current_phase: Optional[str]
    current_ticket: Optional[str]
    current_agent: Optional[str]
    current_step: Optional[str]
    last_completed_step: Optional[str]
    failed_step: Optional[str]
    failure_reason: Optional[str]
    blocked_reason: Optional[str]
    last_run: Optional[str]
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        """Create a PipelineState from a raw JSON dictionary."""
        return cls(
            status=data.get("status", "unknown"),
            current_phase=data.get("current_phase"),
            current_ticket=data.get("current_ticket"),
            current_agent=data.get("current_agent"),
            current_step=data.get("current_step"),
            last_completed_step=data.get("last_completed_step"),
            failed_step=data.get("failed_step"),
            failure_reason=data.get("failure_reason"),
            blocked_reason=data.get("blocked_reason"),
            last_run=data.get("last_run"),
            raw=data,
        )


@dataclass(frozen=True)
class BuildOutput:
    """Typed representation of key fields from build-output.json."""

    status: str
    branch: Optional[str]
    pr_number: Optional[int]
    files_changed: list[str]
    summary: Optional[str]
    ticket_id: Optional[str]
    self_check: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> "BuildOutput":
        """Create a BuildOutput from a raw JSON dictionary."""
        # files_changed can be at different nesting levels depending on
        # the build agent output format.
        files_changed = data.get("files_changed", [])
        if not isinstance(files_changed, list):
            files_changed = []

        return cls(
            status=data.get("status", "unknown"),
            branch=data.get("branch"),
            pr_number=data.get("pr_number"),
            files_changed=files_changed,
            summary=data.get("summary"),
            ticket_id=data.get("ticket_id"),
            self_check=data.get("self_check", {}),
            raw=data,
        )


@dataclass(frozen=True)
class PlanOutput:
    """Typed representation of key fields from plan-output.json."""

    status: str
    next_ticket: Optional[str]
    ticket_id: Optional[str]
    next_ticket_details: Optional[dict]
    backlog_summary: Optional[dict]
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> "PlanOutput":
        """Create a PlanOutput from a raw JSON dictionary."""
        return cls(
            status=data.get("status", "unknown"),
            next_ticket=data.get("next_ticket"),
            ticket_id=data.get("ticket_id"),
            next_ticket_details=data.get("next_ticket_details"),
            backlog_summary=data.get("backlog_summary"),
            raw=data,
        )


@dataclass(frozen=True)
class BacklogTicket:
    """A single ticket extracted from backlog.json."""

    id: str
    github_issue: Optional[int]
    title: str
    description: str
    phase: Optional[str]
    priority: Optional[str]
    type: Optional[str]
    status: str
    dependencies: list[str]
    assigned_agent: Optional[str]
    estimated_complexity: Optional[str]
    pr_number: Optional[int]
    completed_at: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "BacklogTicket":
        """Create a BacklogTicket from a raw ticket dictionary."""
        return cls(
            id=data.get("id", ""),
            github_issue=data.get("github_issue"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            phase=data.get("phase"),
            priority=data.get("priority"),
            type=data.get("type"),
            status=data.get("status", "unknown"),
            dependencies=data.get("dependencies", []),
            assigned_agent=data.get("assigned_agent"),
            estimated_complexity=data.get("estimated_complexity"),
            pr_number=data.get("pr_number"),
            completed_at=data.get("completed_at"),
        )


@dataclass(frozen=True)
class BacklogPhase:
    """A single phase extracted from backlog.json."""

    id: str
    name: str
    description: str
    milestone: str
    status: str
    tickets: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "BacklogPhase":
        """Create a BacklogPhase from a raw phase dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            milestone=data.get("milestone", ""),
            status=data.get("status", "unknown"),
            tickets=data.get("tickets", []),
        )


@dataclass(frozen=True)
class BacklogState:
    """Typed representation of the full backlog.json contents."""

    project: str
    phases: list[BacklogPhase]
    tickets: list[BacklogTicket]
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict) -> "BacklogState":
        """Create a BacklogState from a raw JSON dictionary."""
        phases = [
            BacklogPhase.from_dict(p) for p in data.get("phases", [])
        ]
        tickets = [
            BacklogTicket.from_dict(t) for t in data.get("tickets", [])
        ]
        return cls(
            project=data.get("project", ""),
            phases=phases,
            tickets=tickets,
            raw=data,
        )

    def get_ticket(self, ticket_id: str) -> Optional[BacklogTicket]:
        """Look up a ticket by its ID. Returns None if not found."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None

    def get_phase(self, phase_id: str) -> Optional[BacklogPhase]:
        """Look up a phase by its ID. Returns None if not found."""
        for phase in self.phases:
            if phase.id == phase_id:
                return phase
        return None

    def get_tickets_for_phase(self, phase_id: str) -> list[BacklogTicket]:
        """Return all tickets belonging to a given phase."""
        return [t for t in self.tickets if t.phase == phase_id]

    def get_tickets_by_status(self, status: str) -> list[BacklogTicket]:
        """Return all tickets with a given status."""
        return [t for t in self.tickets if t.status == status]


@dataclass
class TicketContext:
    """Aggregated context about the current ticket from all state files.

    This is the return type of :meth:`StateReader.get_current_ticket_context`.
    It combines information from pipeline.json, build-output.json,
    plan-output.json, and backlog.json into a single convenient object.
    """

    ticket_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    phase: Optional[str] = None
    phase_name: Optional[str] = None
    priority: Optional[str] = None
    complexity: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    pipeline_status: Optional[str] = None
    pipeline_step: Optional[str] = None
    pipeline_agent: Optional[str] = None
    last_completed_step: Optional[str] = None
    branch: Optional[str] = None
    pr_number: Optional[int] = None
    build_status: Optional[str] = None
    files_changed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """An entry in the read cache."""

    data: Any
    read_at: float  # time.monotonic() when the data was read


# ---------------------------------------------------------------------------
# StateReader
# ---------------------------------------------------------------------------


class StateReader:
    """Reads Agent Forge state files with caching and graceful error handling.

    The reader locates the ``.agent-forge/`` directory relative to a given
    project root path.  All file reads are cached in memory: repeated calls
    within the cache TTL return the same data without re-reading from disk.

    Parameters
    ----------
    project_root:
        The root directory of the project that contains ``.agent-forge/``.
        This should be an absolute path.
    cache_ttl_seconds:
        How long cached state file data is considered fresh.  After this
        interval, the next read will re-read from disk.  Set to ``0`` to
        disable caching (every call reads from disk).
    """

    def __init__(
        self,
        project_root: str,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._agent_forge_dir = self._project_root / AGENT_FORGE_DIR
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}

        if not self._agent_forge_dir.is_dir():
            logger.warning(
                "StateReader: .agent-forge directory not found at %s. "
                "State file reads will return None.",
                self._agent_forge_dir,
            )

    # ------------------------------------------------------------------
    # Public API -- Individual readers
    # ------------------------------------------------------------------

    def read_pipeline(self) -> Optional[PipelineState]:
        """Read and parse ``.agent-forge/state/pipeline.json``.

        Returns
        -------
        PipelineState or None
            The parsed pipeline state, or None if the file is missing
            or malformed.
        """
        data = self._read_json(PIPELINE_STATE_PATH)
        if data is None:
            return None
        try:
            return PipelineState.from_dict(data)
        except Exception:
            logger.warning(
                "StateReader: Failed to parse pipeline.json.",
                exc_info=True,
            )
            return None

    def read_build_output(self) -> Optional[BuildOutput]:
        """Read and parse ``.agent-forge/state/build-output.json``.

        Returns
        -------
        BuildOutput or None
            The parsed build output, or None if the file is missing
            or malformed.
        """
        data = self._read_json(BUILD_OUTPUT_PATH)
        if data is None:
            return None
        try:
            return BuildOutput.from_dict(data)
        except Exception:
            logger.warning(
                "StateReader: Failed to parse build-output.json.",
                exc_info=True,
            )
            return None

    def read_plan_output(self) -> Optional[PlanOutput]:
        """Read and parse ``.agent-forge/state/plan-output.json``.

        Returns
        -------
        PlanOutput or None
            The parsed plan output, or None if the file is missing
            or malformed.
        """
        data = self._read_json(PLAN_OUTPUT_PATH)
        if data is None:
            return None
        try:
            return PlanOutput.from_dict(data)
        except Exception:
            logger.warning(
                "StateReader: Failed to parse plan-output.json.",
                exc_info=True,
            )
            return None

    def read_backlog(self) -> Optional[BacklogState]:
        """Read and parse ``.agent-forge/plans/backlog.json``.

        Returns
        -------
        BacklogState or None
            The parsed backlog, or None if the file is missing or malformed.
        """
        data = self._read_json(BACKLOG_PATH)
        if data is None:
            return None
        try:
            return BacklogState.from_dict(data)
        except Exception:
            logger.warning(
                "StateReader: Failed to parse backlog.json.",
                exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Public API -- Convenience aggregator
    # ------------------------------------------------------------------

    def get_current_ticket_context(self) -> Optional[TicketContext]:
        """Aggregate all state files into a single ticket context.

        Reads pipeline.json to determine the current ticket, then enriches
        that with data from backlog.json (ticket details), build-output.json
        (build state), and plan-output.json (plan context).

        Returns
        -------
        TicketContext or None
            A fully populated context object, or None if no ticket is
            currently being processed (pipeline is idle or state files
            are unavailable).
        """
        pipeline = self.read_pipeline()
        if pipeline is None:
            logger.debug(
                "StateReader: Cannot build ticket context -- "
                "pipeline.json not available."
            )
            return None

        # Determine the current ticket ID from pipeline state.
        ticket_id = pipeline.current_ticket
        if ticket_id is None:
            # No ticket currently active -- check plan output for next ticket.
            plan = self.read_plan_output()
            if plan is not None and plan.next_ticket:
                ticket_id = plan.next_ticket
            else:
                logger.debug(
                    "StateReader: No current ticket in pipeline or plan output."
                )
                return None

        ctx = TicketContext(
            ticket_id=ticket_id,
            pipeline_status=pipeline.status,
            pipeline_step=pipeline.current_step,
            pipeline_agent=pipeline.current_agent,
            last_completed_step=pipeline.last_completed_step,
        )

        # Enrich from backlog.
        backlog = self.read_backlog()
        if backlog is not None:
            ticket = backlog.get_ticket(ticket_id)
            if ticket is not None:
                ctx.title = ticket.title
                ctx.description = ticket.description
                ctx.phase = ticket.phase
                ctx.priority = ticket.priority
                ctx.complexity = ticket.estimated_complexity
                ctx.dependencies = list(ticket.dependencies)

            # Look up the phase name.
            if ctx.phase is not None:
                phase = backlog.get_phase(ctx.phase)
                if phase is not None:
                    ctx.phase_name = phase.name

        # Enrich from build output.
        build = self.read_build_output()
        if build is not None and build.ticket_id == ticket_id:
            ctx.branch = build.branch
            ctx.pr_number = build.pr_number
            ctx.build_status = build.status
            ctx.files_changed = list(build.files_changed)

        return ctx

    # ------------------------------------------------------------------
    # Public API -- Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self, file_key: Optional[str] = None) -> None:
        """Invalidate cached state file data.

        Parameters
        ----------
        file_key:
            The relative path key of a specific file to invalidate
            (e.g., ``"state/pipeline.json"``).  If *None*, the entire
            cache is cleared.
        """
        if file_key is not None:
            self._cache.pop(file_key, None)
            logger.debug("StateReader: Invalidated cache for %s.", file_key)
        else:
            self._cache.clear()
            logger.debug("StateReader: Invalidated entire cache.")

    def refresh(self) -> None:
        """Force refresh of all cached state files.

        Equivalent to calling :meth:`invalidate_cache` with no arguments.
        The next read call for each file will re-read from disk.
        """
        self.invalidate_cache()

    @property
    def project_root(self) -> Path:
        """The resolved project root path."""
        return self._project_root

    @property
    def agent_forge_dir(self) -> Path:
        """The resolved .agent-forge directory path."""
        return self._agent_forge_dir

    @property
    def cache_ttl(self) -> float:
        """The cache TTL in seconds."""
        return self._cache_ttl

    @cache_ttl.setter
    def cache_ttl(self, value: float) -> None:
        """Set the cache TTL.  Must be non-negative."""
        if value < 0:
            raise ValueError(
                f"cache_ttl must be non-negative, got {value}."
            )
        self._cache_ttl = value

    def cached_keys(self) -> list[str]:
        """Return a list of currently cached file keys."""
        return list(self._cache.keys())

    def is_cached(self, file_key: str) -> bool:
        """Check whether a given file key has a valid (non-expired) cache entry."""
        entry = self._cache.get(file_key)
        if entry is None:
            return False
        elapsed = time.monotonic() - entry.read_at
        return elapsed < self._cache_ttl

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_json(self, relative_path: str) -> Optional[dict]:
        """Read a JSON file from .agent-forge/ with caching.

        Parameters
        ----------
        relative_path:
            Path relative to .agent-forge/ (e.g., "state/pipeline.json").

        Returns
        -------
        dict or None
            The parsed JSON data, or None if the file does not exist,
            is not valid JSON, or cannot be read.
        """
        # Check cache first.
        entry = self._cache.get(relative_path)
        if entry is not None:
            elapsed = time.monotonic() - entry.read_at
            if elapsed < self._cache_ttl:
                logger.debug(
                    "StateReader: Returning cached %s (age=%.1fs).",
                    relative_path,
                    elapsed,
                )
                return entry.data

        # Read from disk.
        file_path = self._agent_forge_dir / relative_path
        data = self._safe_read_json_file(file_path)

        # Update cache regardless of success -- None is also cached to avoid
        # repeated disk hits for missing files within the TTL window.
        self._cache[relative_path] = _CacheEntry(
            data=data,
            read_at=time.monotonic(),
        )

        return data

    @staticmethod
    def _safe_read_json_file(path: Path) -> Optional[dict]:
        """Read and parse a JSON file, returning None on any error.

        Handles missing files, permission errors, and malformed JSON
        without raising exceptions.
        """
        if not path.is_file():
            logger.debug(
                "StateReader: File not found: %s.",
                path,
            )
            return None

        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if not isinstance(data, dict):
                logger.warning(
                    "StateReader: %s does not contain a JSON object. "
                    "Got %s.",
                    path,
                    type(data).__name__,
                )
                return None
            return data
        except json.JSONDecodeError:
            logger.warning(
                "StateReader: %s contains invalid JSON.",
                path,
                exc_info=True,
            )
            return None
        except OSError:
            logger.warning(
                "StateReader: Could not read %s.",
                path,
                exc_info=True,
            )
            return None
