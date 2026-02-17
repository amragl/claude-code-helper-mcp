"""Configuration and settings module for Claude Code Helper MCP.

Provides the :class:`MemoryConfig` class which centralises all configuration
for the memory system.  Configuration is resolved in priority order:

1. **Environment variables** (highest priority) -- ``CLAUDE_MEMORY_*``
2. **Config file** -- ``<project_root>/.claude-memory/config.json``
3. **Defaults** (lowest priority) -- sensible built-in values

Typical usage::

    config = MemoryConfig.load()                          # auto-detect project root
    config = MemoryConfig.load("/path/to/project")        # explicit project root
    config = MemoryConfig(storage_path="/custom/path")    # programmatic construction

    print(config.storage_path)   # resolved absolute path to .claude-memory/
    print(config.window_size)    # 3  (or overridden value)
    print(config.log_level)      # "INFO"  (or overridden value)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default storage directory name, placed at the project root.
DEFAULT_STORAGE_DIR_NAME = ".claude-memory"

# Config file name inside the storage directory.
CONFIG_FILE_NAME = "config.json"

# Environment variable prefix.  All config keys can be overridden by setting
# ``CLAUDE_MEMORY_<UPPER_KEY>``.  For example, ``CLAUDE_MEMORY_WINDOW_SIZE=5``.
ENV_PREFIX = "CLAUDE_MEMORY_"

# Sentinel files used to detect a project root directory.  The search walks
# upward from the current working directory until one of these is found.
PROJECT_ROOT_MARKERS = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    ".claude-memory",
)

# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class MemoryConfig(BaseModel):
    """Centralised configuration for the Claude Code Helper memory system.

    Every field has a sensible default.  Fields can be overridden by a
    ``config.json`` file or by environment variables (see module docstring).

    Attributes
    ----------
    storage_path:
        Absolute path to the storage directory.  When not set explicitly, it
        is derived from the detected project root + :data:`DEFAULT_STORAGE_DIR_NAME`.
    window_size:
        Maximum number of completed tasks to retain in the sliding window
        (not counting the current active task).  Must be >= 1 and <= 100.
    log_level:
        Python logging level name.  One of ``DEBUG``, ``INFO``, ``WARNING``,
        ``ERROR``, ``CRITICAL``.
    auto_save:
        Whether to automatically persist task state after each mutation.
        Disabling this allows batch updates with explicit saves.
    archive_completed:
        Whether to keep individual task JSON files on disk after tasks are
        archived out of the sliding window.  When *False*, archived task
        files are deleted to save space.
    project_root:
        The detected or configured project root path.  This is used to
        derive ``storage_path`` when it is not set explicitly.  Read-only
        after construction.
    """

    storage_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the .claude-memory/ storage directory.",
    )
    window_size: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Max completed tasks in the sliding window (1-100).",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    auto_save: bool = Field(
        default=True,
        description="Auto-persist state after each task mutation.",
    )
    archive_completed: bool = Field(
        default=True,
        description="Keep archived task files on disk after window rotation.",
    )
    project_root: Optional[str] = Field(
        default=None,
        description="Detected or configured project root path.",
    )

    # Usage burn tracking settings.
    usage_tracking_enabled: bool = Field(
        default=True,
        description="Enable usage burn detection and alerting.",
    )
    usage_tool_call_warn: int = Field(
        default=50,
        ge=1,
        description="Warning threshold for tool calls per task.",
    )
    usage_tool_call_critical: int = Field(
        default=100,
        ge=1,
        description="Critical threshold for tool calls per task.",
    )
    usage_step_warn: int = Field(
        default=30,
        ge=1,
        description="Warning threshold for steps per task.",
    )
    usage_step_critical: int = Field(
        default=60,
        ge=1,
        description="Critical threshold for steps per task.",
    )
    usage_time_warn_minutes: float = Field(
        default=15.0,
        gt=0,
        description="Warning threshold for task elapsed time (minutes).",
    )
    usage_time_critical_minutes: float = Field(
        default=30.0,
        gt=0,
        description="Critical threshold for task elapsed time (minutes).",
    )
    usage_burst_critical: int = Field(
        default=10,
        ge=1,
        description="Tool calls within burst window to trigger burst alert.",
    )
    usage_burst_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Sliding window size for burst detection (seconds).",
    )
    usage_session_total_calls_critical: int = Field(
        default=500,
        ge=1,
        description="Critical threshold for cumulative session tool calls.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def resolve_paths(self) -> "MemoryConfig":
        """Resolve ``storage_path`` and ``project_root`` to absolute paths.

        If ``project_root`` is not set, attempt to detect it from the
        current working directory.  If ``storage_path`` is not set, derive
        it from ``project_root``.
        """
        # Resolve project root.
        if self.project_root is not None:
            self.project_root = str(Path(self.project_root).resolve())
        else:
            detected = _detect_project_root()
            if detected is not None:
                self.project_root = str(detected)
            else:
                # Fall back to the current working directory.
                self.project_root = str(Path.cwd())

        # Resolve storage path.
        if self.storage_path is not None:
            self.storage_path = str(Path(self.storage_path).resolve())
        else:
            self.storage_path = str(
                Path(self.project_root) / DEFAULT_STORAGE_DIR_NAME
            )

        return self

    @model_validator(mode="after")
    def validate_log_level(self) -> "MemoryConfig":
        """Normalise and validate the log level string."""
        normalised = self.log_level.upper().strip()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalised not in valid_levels:
            raise ValueError(
                f"Invalid log_level '{self.log_level}'. "
                f"Must be one of: {', '.join(sorted(valid_levels))}."
            )
        self.log_level = normalised
        return self

    # ------------------------------------------------------------------
    # Factory: load from file + environment
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        project_root: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> "MemoryConfig":
        """Load configuration with full resolution: file -> env -> defaults.

        Parameters
        ----------
        project_root:
            Explicit project root.  When *None*, auto-detection is used.
        config_path:
            Explicit path to a ``config.json`` file.  When *None*, the
            config file is looked up at
            ``<project_root>/.claude-memory/config.json``.

        Returns
        -------
        MemoryConfig
            Fully resolved configuration object.
        """
        # Step 1: Determine the project root (for finding the config file).
        resolved_root: Optional[str] = None
        if project_root is not None:
            resolved_root = str(Path(project_root).resolve())
        else:
            detected = _detect_project_root()
            if detected is not None:
                resolved_root = str(detected)
            else:
                resolved_root = str(Path.cwd())

        # Step 2: Load values from config file (if it exists).
        file_values = _load_config_file(resolved_root, config_path)

        # Step 3: Apply environment variable overrides.
        env_values = _load_env_overrides()

        # Step 4: Merge in priority order: env > file > defaults.
        # Start with file values (if any), then overlay env values.
        merged: dict = {}
        merged["project_root"] = resolved_root
        if file_values:
            merged.update(file_values)
        if env_values:
            merged.update(env_values)
        # Ensure project_root is always the explicitly resolved one.
        merged["project_root"] = resolved_root

        return cls.model_validate(merged)

    # ------------------------------------------------------------------
    # Persistence: save config to disk
    # ------------------------------------------------------------------

    def save(self, config_path: Optional[str] = None) -> Path:
        """Save the current configuration to a JSON file.

        Parameters
        ----------
        config_path:
            Explicit file path.  When *None*, writes to
            ``<storage_path>/config.json``.

        Returns
        -------
        Path
            The path to the written file.
        """
        if config_path is not None:
            target = Path(config_path).resolve()
        else:
            target = Path(self.storage_path) / CONFIG_FILE_NAME

        target.parent.mkdir(parents=True, exist_ok=True)

        # Serialise only the user-facing fields (not project_root, which is
        # derived).  This keeps the config file clean and portable.
        data = {
            "window_size": self.window_size,
            "log_level": self.log_level,
            "auto_save": self.auto_save,
            "archive_completed": self.archive_completed,
        }
        # Only include storage_path if it was explicitly set (not derived).
        default_storage = str(
            Path(self.project_root) / DEFAULT_STORAGE_DIR_NAME
        )
        if self.storage_path != default_storage:
            data["storage_path"] = self.storage_path

        with open(target, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2, ensure_ascii=False)
            fp.write("\n")

        logger.info("Saved configuration to %s", target)
        return target

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def configure_logging(self) -> None:
        """Apply the configured log level to the ``claude_code_helper_mcp`` logger.

        Sets up a basic logging configuration with the configured level.
        This is idempotent -- calling it multiple times is safe.
        """
        pkg_logger = logging.getLogger("claude_code_helper_mcp")
        pkg_logger.setLevel(self.log_level)

        # Add a StreamHandler if the logger has no handlers yet.
        if not pkg_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self.log_level)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            pkg_logger.addHandler(handler)

    def to_dict(self) -> dict:
        """Return all configuration values as a plain dictionary."""
        return self.model_dump()

    def __repr__(self) -> str:
        return (
            f"MemoryConfig("
            f"storage_path={self.storage_path!r}, "
            f"window_size={self.window_size}, "
            f"log_level={self.log_level!r}, "
            f"auto_save={self.auto_save}, "
            f"archive_completed={self.archive_completed}, "
            f"project_root={self.project_root!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from *start_path* to find the project root.

    The project root is the first directory that contains one of the
    :data:`PROJECT_ROOT_MARKERS`.  If no marker is found before reaching
    the filesystem root, returns *None*.

    Parameters
    ----------
    start_path:
        The directory to start searching from.  Defaults to ``os.getcwd()``.
    """
    current = (start_path or Path.cwd()).resolve()

    # Safety limit to prevent infinite loops on unusual filesystems.
    max_depth = 50
    for _ in range(max_depth):
        for marker in PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current

        parent = current.parent
        if parent == current:
            # Reached the filesystem root.
            break
        current = parent

    return None


def _load_config_file(
    project_root: str,
    config_path: Optional[str] = None,
) -> dict:
    """Read a ``config.json`` file and return its contents as a dict.

    Returns an empty dict if the file does not exist or is malformed.
    """
    if config_path is not None:
        path = Path(config_path).resolve()
    else:
        path = Path(project_root) / DEFAULT_STORAGE_DIR_NAME / CONFIG_FILE_NAME

    if not path.is_file():
        logger.debug("No config file at %s. Using defaults.", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            logger.warning(
                "Config file %s does not contain a JSON object. Ignoring.",
                path,
            )
            return {}
        logger.info("Loaded configuration from %s", path)
        return data
    except json.JSONDecodeError:
        logger.warning(
            "Config file %s contains invalid JSON. Ignoring.",
            path,
            exc_info=True,
        )
        return {}
    except OSError:
        logger.warning(
            "Could not read config file %s. Ignoring.",
            path,
            exc_info=True,
        )
        return {}


def _load_env_overrides() -> dict:
    """Read ``CLAUDE_MEMORY_*`` environment variables and return overrides.

    Supported variables (case-insensitive after the prefix):

    - ``CLAUDE_MEMORY_STORAGE_PATH`` -- override storage_path
    - ``CLAUDE_MEMORY_WINDOW_SIZE`` -- override window_size (integer)
    - ``CLAUDE_MEMORY_LOG_LEVEL`` -- override log_level
    - ``CLAUDE_MEMORY_AUTO_SAVE`` -- override auto_save (``true``/``false``)
    - ``CLAUDE_MEMORY_ARCHIVE_COMPLETED`` -- override archive_completed

    Returns a dict of field_name -> value for any variables that are set.
    """
    overrides: dict = {}

    storage_path = os.environ.get(f"{ENV_PREFIX}STORAGE_PATH")
    if storage_path is not None:
        overrides["storage_path"] = storage_path

    window_size = os.environ.get(f"{ENV_PREFIX}WINDOW_SIZE")
    if window_size is not None:
        try:
            overrides["window_size"] = int(window_size)
        except ValueError:
            logger.warning(
                "Invalid %sWINDOW_SIZE value: %r. Must be an integer. Ignoring.",
                ENV_PREFIX,
                window_size,
            )

    log_level = os.environ.get(f"{ENV_PREFIX}LOG_LEVEL")
    if log_level is not None:
        overrides["log_level"] = log_level

    auto_save = os.environ.get(f"{ENV_PREFIX}AUTO_SAVE")
    if auto_save is not None:
        overrides["auto_save"] = auto_save.lower() in ("true", "1", "yes")

    archive_completed = os.environ.get(f"{ENV_PREFIX}ARCHIVE_COMPLETED")
    if archive_completed is not None:
        overrides["archive_completed"] = archive_completed.lower() in (
            "true",
            "1",
            "yes",
        )

    # Usage tracking overrides.
    usage_enabled = os.environ.get(f"{ENV_PREFIX}USAGE_TRACKING_ENABLED")
    if usage_enabled is not None:
        overrides["usage_tracking_enabled"] = usage_enabled.lower() in (
            "true", "1", "yes",
        )

    _int_usage_keys = {
        "USAGE_TOOL_CALL_WARN": "usage_tool_call_warn",
        "USAGE_TOOL_CALL_CRITICAL": "usage_tool_call_critical",
        "USAGE_STEP_WARN": "usage_step_warn",
        "USAGE_STEP_CRITICAL": "usage_step_critical",
        "USAGE_BURST_CRITICAL": "usage_burst_critical",
        "USAGE_BURST_WINDOW_SECONDS": "usage_burst_window_seconds",
        "USAGE_SESSION_TOTAL_CALLS_CRITICAL": "usage_session_total_calls_critical",
    }
    for env_key, field_name in _int_usage_keys.items():
        val = os.environ.get(f"{ENV_PREFIX}{env_key}")
        if val is not None:
            try:
                overrides[field_name] = int(val)
            except ValueError:
                logger.warning(
                    "Invalid %s%s value: %r. Must be an integer. Ignoring.",
                    ENV_PREFIX, env_key, val,
                )

    _float_usage_keys = {
        "USAGE_TIME_WARN_MINUTES": "usage_time_warn_minutes",
        "USAGE_TIME_CRITICAL_MINUTES": "usage_time_critical_minutes",
    }
    for env_key, field_name in _float_usage_keys.items():
        val = os.environ.get(f"{ENV_PREFIX}{env_key}")
        if val is not None:
            try:
                overrides[field_name] = float(val)
            except ValueError:
                logger.warning(
                    "Invalid %s%s value: %r. Must be a number. Ignoring.",
                    ENV_PREFIX, env_key, val,
                )

    if overrides:
        logger.info(
            "Environment overrides applied: %s",
            ", ".join(overrides.keys()),
        )

    return overrides
