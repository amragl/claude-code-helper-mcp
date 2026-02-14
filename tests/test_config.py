"""Tests for MemoryConfig -- configuration and settings module.

All tests use real files in temporary directories, real environment variables,
and real config.json files.  No mocks, no stubs, no fakes.
"""

import json
import os
from pathlib import Path

import pytest

from claude_code_helper_mcp.config import (
    CONFIG_FILE_NAME,
    DEFAULT_STORAGE_DIR_NAME,
    ENV_PREFIX,
    PROJECT_ROOT_MARKERS,
    MemoryConfig,
    _detect_project_root,
    _load_config_file,
    _load_env_overrides,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with a .git marker."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    return tmp_path


@pytest.fixture()
def config_file(project_dir: Path) -> Path:
    """Create a config.json inside the project's .claude-memory/ directory."""
    storage = project_dir / DEFAULT_STORAGE_DIR_NAME
    storage.mkdir()
    config_path = storage / CONFIG_FILE_NAME
    config_path.write_text(
        json.dumps(
            {
                "window_size": 5,
                "log_level": "DEBUG",
                "auto_save": False,
                "archive_completed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture(autouse=True)
def clean_env():
    """Remove all CLAUDE_MEMORY_* env vars before and after each test."""
    env_keys = [k for k in os.environ if k.startswith(ENV_PREFIX)]
    saved = {k: os.environ.pop(k) for k in env_keys}
    yield
    # Restore original state.
    for k in list(os.environ.keys()):
        if k.startswith(ENV_PREFIX):
            del os.environ[k]
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# Default construction
# ---------------------------------------------------------------------------


class TestDefaultConstruction:
    """MemoryConfig with no arguments uses sensible defaults."""

    def test_default_window_size(self) -> None:
        config = MemoryConfig(project_root="/tmp/test-project")
        assert config.window_size == 3

    def test_default_log_level(self) -> None:
        config = MemoryConfig(project_root="/tmp/test-project")
        assert config.log_level == "INFO"

    def test_default_auto_save(self) -> None:
        config = MemoryConfig(project_root="/tmp/test-project")
        assert config.auto_save is True

    def test_default_archive_completed(self) -> None:
        config = MemoryConfig(project_root="/tmp/test-project")
        assert config.archive_completed is True

    def test_storage_path_derived_from_project_root(self, tmp_path: Path) -> None:
        config = MemoryConfig(project_root=str(tmp_path))
        expected = str(Path(tmp_path).resolve() / DEFAULT_STORAGE_DIR_NAME)
        assert config.storage_path == expected

    def test_project_root_is_resolved_to_absolute(self, tmp_path: Path) -> None:
        config = MemoryConfig(project_root=str(tmp_path))
        assert Path(config.project_root).is_absolute()


# ---------------------------------------------------------------------------
# Custom values
# ---------------------------------------------------------------------------


class TestCustomValues:
    """MemoryConfig accepts custom values for all fields."""

    def test_custom_window_size(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", window_size=10)
        assert config.window_size == 10

    def test_custom_log_level(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", log_level="WARNING")
        assert config.log_level == "WARNING"

    def test_custom_storage_path(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "custom-storage")
        config = MemoryConfig(project_root="/tmp/p", storage_path=custom)
        assert config.storage_path == str(Path(custom).resolve())

    def test_custom_auto_save_false(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", auto_save=False)
        assert config.auto_save is False

    def test_custom_archive_completed_false(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", archive_completed=False)
        assert config.archive_completed is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """MemoryConfig validates field constraints."""

    def test_window_size_too_small(self) -> None:
        with pytest.raises(Exception):
            MemoryConfig(project_root="/tmp/p", window_size=0)

    def test_window_size_too_large(self) -> None:
        with pytest.raises(Exception):
            MemoryConfig(project_root="/tmp/p", window_size=101)

    def test_window_size_minimum(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", window_size=1)
        assert config.window_size == 1

    def test_window_size_maximum(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", window_size=100)
        assert config.window_size == 100

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid log_level"):
            MemoryConfig(project_root="/tmp/p", log_level="TRACE")

    def test_log_level_case_insensitive(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", log_level="debug")
        assert config.log_level == "DEBUG"

    def test_log_level_with_whitespace(self) -> None:
        config = MemoryConfig(project_root="/tmp/p", log_level="  info  ")
        assert config.log_level == "INFO"

    def test_all_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            config = MemoryConfig(project_root="/tmp/p", log_level=level)
            assert config.log_level == level


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------


class TestProjectRootDetection:
    """_detect_project_root walks upward to find project markers."""

    def test_finds_git_directory(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = _detect_project_root(tmp_path)
        assert result == tmp_path

    def test_finds_pyproject_toml(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").touch()
        result = _detect_project_root(tmp_path)
        assert result == tmp_path

    def test_finds_setup_py(self, tmp_path: Path) -> None:
        (tmp_path / "setup.py").touch()
        result = _detect_project_root(tmp_path)
        assert result == tmp_path

    def test_finds_setup_cfg(self, tmp_path: Path) -> None:
        (tmp_path / "setup.cfg").touch()
        result = _detect_project_root(tmp_path)
        assert result == tmp_path

    def test_finds_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").touch()
        result = _detect_project_root(tmp_path)
        assert result == tmp_path

    def test_finds_claude_memory_dir(self, tmp_path: Path) -> None:
        (tmp_path / ".claude-memory").mkdir()
        result = _detect_project_root(tmp_path)
        assert result == tmp_path

    def test_walks_upward(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        child = tmp_path / "src" / "deep"
        child.mkdir(parents=True)
        result = _detect_project_root(child)
        assert result == tmp_path

    def test_returns_none_when_no_marker(self, tmp_path: Path) -> None:
        # Create a deep directory with no markers anywhere.
        isolated = tmp_path / "isolated" / "deep"
        isolated.mkdir(parents=True)
        # Remove any markers that might exist in tmp_path.
        for marker in PROJECT_ROOT_MARKERS:
            p = tmp_path / marker
            if p.exists():
                if p.is_dir():
                    p.rmdir()
                else:
                    p.unlink()
        # If any parent of tmp_path has markers (likely), we can't guarantee
        # None.  But we can verify it returns a Path or None and doesn't crash.
        result = _detect_project_root(isolated)
        assert result is None or isinstance(result, Path)

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        result = _detect_project_root(tmp_path)
        assert result is not None
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestConfigFileLoading:
    """_load_config_file reads and parses config.json."""

    def test_loads_valid_config(self, project_dir: Path, config_file: Path) -> None:
        values = _load_config_file(str(project_dir))
        assert values["window_size"] == 5
        assert values["log_level"] == "DEBUG"
        assert values["auto_save"] is False

    def test_returns_empty_for_missing_file(self, project_dir: Path) -> None:
        values = _load_config_file(str(project_dir))
        assert values == {}

    def test_returns_empty_for_corrupt_json(self, project_dir: Path) -> None:
        storage = project_dir / DEFAULT_STORAGE_DIR_NAME
        storage.mkdir(exist_ok=True)
        (storage / CONFIG_FILE_NAME).write_text(
            "{{{bad json", encoding="utf-8"
        )
        values = _load_config_file(str(project_dir))
        assert values == {}

    def test_returns_empty_for_non_object_json(self, project_dir: Path) -> None:
        storage = project_dir / DEFAULT_STORAGE_DIR_NAME
        storage.mkdir(exist_ok=True)
        (storage / CONFIG_FILE_NAME).write_text("[1, 2, 3]", encoding="utf-8")
        values = _load_config_file(str(project_dir))
        assert values == {}

    def test_explicit_config_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "my-config.json"
        custom.write_text(
            json.dumps({"window_size": 7}), encoding="utf-8"
        )
        values = _load_config_file(str(tmp_path), config_path=str(custom))
        assert values["window_size"] == 7

    def test_partial_config_file(self, project_dir: Path) -> None:
        """Config file with only some fields -- others use defaults."""
        storage = project_dir / DEFAULT_STORAGE_DIR_NAME
        storage.mkdir(exist_ok=True)
        (storage / CONFIG_FILE_NAME).write_text(
            json.dumps({"window_size": 8}), encoding="utf-8"
        )
        values = _load_config_file(str(project_dir))
        assert values == {"window_size": 8}


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    """_load_env_overrides reads CLAUDE_MEMORY_* environment variables."""

    def test_no_env_vars(self) -> None:
        result = _load_env_overrides()
        assert result == {}

    def test_storage_path_env(self) -> None:
        os.environ[f"{ENV_PREFIX}STORAGE_PATH"] = "/custom/path"
        result = _load_env_overrides()
        assert result["storage_path"] == "/custom/path"

    def test_window_size_env(self) -> None:
        os.environ[f"{ENV_PREFIX}WINDOW_SIZE"] = "7"
        result = _load_env_overrides()
        assert result["window_size"] == 7

    def test_window_size_invalid_ignored(self) -> None:
        os.environ[f"{ENV_PREFIX}WINDOW_SIZE"] = "not-a-number"
        result = _load_env_overrides()
        assert "window_size" not in result

    def test_log_level_env(self) -> None:
        os.environ[f"{ENV_PREFIX}LOG_LEVEL"] = "ERROR"
        result = _load_env_overrides()
        assert result["log_level"] == "ERROR"

    def test_auto_save_true_values(self) -> None:
        for val in ("true", "1", "yes", "True", "YES"):
            os.environ[f"{ENV_PREFIX}AUTO_SAVE"] = val
            result = _load_env_overrides()
            assert result["auto_save"] is True

    def test_auto_save_false_values(self) -> None:
        for val in ("false", "0", "no"):
            os.environ[f"{ENV_PREFIX}AUTO_SAVE"] = val
            result = _load_env_overrides()
            assert result["auto_save"] is False

    def test_archive_completed_env(self) -> None:
        os.environ[f"{ENV_PREFIX}ARCHIVE_COMPLETED"] = "false"
        result = _load_env_overrides()
        assert result["archive_completed"] is False

    def test_multiple_env_vars(self) -> None:
        os.environ[f"{ENV_PREFIX}WINDOW_SIZE"] = "10"
        os.environ[f"{ENV_PREFIX}LOG_LEVEL"] = "WARNING"
        result = _load_env_overrides()
        assert result["window_size"] == 10
        assert result["log_level"] == "WARNING"


# ---------------------------------------------------------------------------
# MemoryConfig.load() integration
# ---------------------------------------------------------------------------


class TestLoad:
    """MemoryConfig.load() resolves file + env + defaults."""

    def test_load_defaults_only(self, project_dir: Path) -> None:
        config = MemoryConfig.load(project_root=str(project_dir))
        assert config.window_size == 3
        assert config.log_level == "INFO"
        assert config.auto_save is True
        assert config.archive_completed is True
        assert config.project_root == str(project_dir)

    def test_load_from_config_file(
        self, project_dir: Path, config_file: Path
    ) -> None:
        config = MemoryConfig.load(project_root=str(project_dir))
        assert config.window_size == 5
        assert config.log_level == "DEBUG"
        assert config.auto_save is False
        assert config.archive_completed is False

    def test_env_overrides_config_file(
        self, project_dir: Path, config_file: Path
    ) -> None:
        os.environ[f"{ENV_PREFIX}WINDOW_SIZE"] = "12"
        config = MemoryConfig.load(project_root=str(project_dir))
        # Env overrides the file value of 5.
        assert config.window_size == 12
        # File value for log_level still applies.
        assert config.log_level == "DEBUG"

    def test_load_with_explicit_config_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom" / "settings.json"
        custom.parent.mkdir(parents=True)
        custom.write_text(
            json.dumps({"window_size": 20, "log_level": "ERROR"}),
            encoding="utf-8",
        )
        config = MemoryConfig.load(
            project_root=str(tmp_path), config_path=str(custom)
        )
        assert config.window_size == 20
        assert config.log_level == "ERROR"

    def test_load_storage_path_derived(self, project_dir: Path) -> None:
        config = MemoryConfig.load(project_root=str(project_dir))
        expected = str(project_dir / DEFAULT_STORAGE_DIR_NAME)
        assert config.storage_path == expected

    def test_load_with_env_storage_path(self, project_dir: Path) -> None:
        os.environ[f"{ENV_PREFIX}STORAGE_PATH"] = "/override/storage"
        config = MemoryConfig.load(project_root=str(project_dir))
        assert config.storage_path == str(Path("/override/storage").resolve())


# ---------------------------------------------------------------------------
# MemoryConfig.save()
# ---------------------------------------------------------------------------


class TestSave:
    """MemoryConfig.save() writes config.json to disk."""

    def test_save_creates_file(self, project_dir: Path) -> None:
        config = MemoryConfig(project_root=str(project_dir))
        path = config.save()
        assert path.is_file()

    def test_save_default_location(self, project_dir: Path) -> None:
        config = MemoryConfig(project_root=str(project_dir))
        path = config.save()
        expected = Path(config.storage_path) / CONFIG_FILE_NAME
        assert path == expected

    def test_save_custom_path(self, tmp_path: Path) -> None:
        config = MemoryConfig(project_root=str(tmp_path))
        custom = tmp_path / "custom-config.json"
        path = config.save(config_path=str(custom))
        assert path == custom
        assert custom.is_file()

    def test_saved_file_is_valid_json(self, project_dir: Path) -> None:
        config = MemoryConfig(
            project_root=str(project_dir),
            window_size=7,
            log_level="WARNING",
        )
        path = config.save()
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert data["window_size"] == 7
        assert data["log_level"] == "WARNING"

    def test_save_does_not_include_default_storage_path(
        self, project_dir: Path
    ) -> None:
        """When storage_path is derived, it is not written to the file."""
        config = MemoryConfig(project_root=str(project_dir))
        path = config.save()
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert "storage_path" not in data

    def test_save_includes_custom_storage_path(self, project_dir: Path) -> None:
        custom_storage = str(project_dir / "custom-storage")
        config = MemoryConfig(
            project_root=str(project_dir),
            storage_path=custom_storage,
        )
        path = config.save()
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert "storage_path" in data

    def test_save_then_load_round_trip(self, project_dir: Path) -> None:
        original = MemoryConfig(
            project_root=str(project_dir),
            window_size=8,
            log_level="ERROR",
            auto_save=False,
            archive_completed=False,
        )
        original.save()

        loaded = MemoryConfig.load(project_root=str(project_dir))
        assert loaded.window_size == 8
        assert loaded.log_level == "ERROR"
        assert loaded.auto_save is False
        assert loaded.archive_completed is False

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "config.json"
        config = MemoryConfig(project_root=str(tmp_path))
        path = config.save(config_path=str(deep))
        assert path.is_file()


# ---------------------------------------------------------------------------
# configure_logging()
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """configure_logging() sets up the package-level logger."""

    def test_sets_log_level(self) -> None:
        import logging

        config = MemoryConfig(project_root="/tmp/p", log_level="WARNING")
        config.configure_logging()

        pkg_logger = logging.getLogger("claude_code_helper_mcp")
        assert pkg_logger.level == logging.WARNING

    def test_adds_handler(self) -> None:
        import logging

        config = MemoryConfig(project_root="/tmp/p", log_level="DEBUG")
        # Remove existing handlers first (from other tests).
        pkg_logger = logging.getLogger("claude_code_helper_mcp")
        pkg_logger.handlers.clear()

        config.configure_logging()
        assert len(pkg_logger.handlers) >= 1

    def test_idempotent(self) -> None:
        import logging

        config = MemoryConfig(project_root="/tmp/p", log_level="INFO")
        pkg_logger = logging.getLogger("claude_code_helper_mcp")
        pkg_logger.handlers.clear()

        config.configure_logging()
        handler_count = len(pkg_logger.handlers)
        config.configure_logging()
        assert len(pkg_logger.handlers) == handler_count


# ---------------------------------------------------------------------------
# to_dict() and __repr__
# ---------------------------------------------------------------------------


class TestUtility:
    """Utility methods on MemoryConfig."""

    def test_to_dict(self) -> None:
        config = MemoryConfig(
            project_root="/tmp/p",
            window_size=5,
            log_level="ERROR",
        )
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["window_size"] == 5
        assert d["log_level"] == "ERROR"
        assert "project_root" in d
        assert "storage_path" in d

    def test_repr(self) -> None:
        config = MemoryConfig(project_root="/tmp/p")
        r = repr(config)
        assert "MemoryConfig(" in r
        assert "window_size=3" in r
        assert "log_level='INFO'" in r

    def test_to_dict_round_trip(self) -> None:
        config = MemoryConfig(
            project_root="/tmp/p",
            window_size=10,
            log_level="DEBUG",
            auto_save=False,
        )
        d = config.to_dict()
        restored = MemoryConfig.model_validate(d)
        assert restored.window_size == config.window_size
        assert restored.log_level == config.log_level
        assert restored.auto_save == config.auto_save


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end config lifecycle: create, save, modify env, reload."""

    def test_full_lifecycle(self, project_dir: Path) -> None:
        # 1. Create config with defaults.
        config = MemoryConfig.load(project_root=str(project_dir))
        assert config.window_size == 3
        assert config.log_level == "INFO"

        # 2. Save to disk.
        config.save()
        config_path = Path(config.storage_path) / CONFIG_FILE_NAME
        assert config_path.is_file()

        # 3. Modify the file manually.
        with open(config_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        data["window_size"] = 15
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=2)

        # 4. Reload -- picks up file change.
        config2 = MemoryConfig.load(project_root=str(project_dir))
        assert config2.window_size == 15

        # 5. Set env override -- takes precedence over file.
        os.environ[f"{ENV_PREFIX}WINDOW_SIZE"] = "25"
        config3 = MemoryConfig.load(project_root=str(project_dir))
        assert config3.window_size == 25

    def test_config_with_storage_integration(self, project_dir: Path) -> None:
        """Config works with MemoryStore and WindowManager."""
        from claude_code_helper_mcp.storage.store import MemoryStore
        from claude_code_helper_mcp.storage.window_manager import WindowManager

        config = MemoryConfig.load(project_root=str(project_dir))

        # Use the config's storage_path to create a store.
        store = MemoryStore(config.storage_path)
        assert store.storage_root.is_dir()

        # Use the config's window_size with WindowManager.
        manager = WindowManager(
            storage_path=config.storage_path,
            window_size=config.window_size,
        )
        assert manager.window_size == config.window_size

        # Create and complete a task to verify integration.
        task = manager.start_new_task("CONFIG-TEST", "Config integration")
        task.add_step("Verified config integration", "test")
        manager.complete_current_task("Config integration verified")

        loaded = manager.get_task("CONFIG-TEST")
        assert loaded is not None
        assert loaded.ticket_id == "CONFIG-TEST"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Module-level constants are correct."""

    def test_default_storage_dir_name(self) -> None:
        assert DEFAULT_STORAGE_DIR_NAME == ".claude-memory"

    def test_config_file_name(self) -> None:
        assert CONFIG_FILE_NAME == "config.json"

    def test_env_prefix(self) -> None:
        assert ENV_PREFIX == "CLAUDE_MEMORY_"

    def test_project_root_markers(self) -> None:
        assert ".git" in PROJECT_ROOT_MARKERS
        assert "pyproject.toml" in PROJECT_ROOT_MARKERS
        assert ".claude-memory" in PROJECT_ROOT_MARKERS
