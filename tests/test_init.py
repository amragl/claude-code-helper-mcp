"""Tests for package initialization and basic imports."""


def test_package_imports():
    """Verify the main package can be imported."""
    import claude_code_helper_mcp

    assert claude_code_helper_mcp is not None


def test_version_defined():
    """Verify __version__ is set and follows semver format."""
    from claude_code_helper_mcp import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) == 3, f"Expected semver (X.Y.Z), got {__version__}"
    for part in parts:
        assert part.isdigit(), f"Version part '{part}' is not a digit in {__version__}"


def test_version_value():
    """Verify the initial version is 0.1.0."""
    from claude_code_helper_mcp import __version__

    assert __version__ == "0.1.0"


def test_subpackages_importable():
    """Verify all subpackages can be imported."""
    import claude_code_helper_mcp.models
    import claude_code_helper_mcp.storage
    import claude_code_helper_mcp.mcp
    import claude_code_helper_mcp.cli
    import claude_code_helper_mcp.detection
    import claude_code_helper_mcp.hooks

    assert claude_code_helper_mcp.models is not None
    assert claude_code_helper_mcp.storage is not None
    assert claude_code_helper_mcp.mcp is not None
    assert claude_code_helper_mcp.cli is not None
    assert claude_code_helper_mcp.detection is not None
    assert claude_code_helper_mcp.hooks is not None


def test_cli_group_exists():
    """Verify the Click CLI group can be imported."""
    from claude_code_helper_mcp.cli.main import cli

    assert cli is not None
    assert callable(cli)
