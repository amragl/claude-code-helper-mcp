"""Main Click CLI entry point for the memory command."""

import click

from claude_code_helper_mcp import __version__


@click.group()
@click.version_option(version=__version__, prog_name="claude-code-helper-mcp")
def cli():
    """Claude Code Helper - Memory management for Claude Code sessions."""


if __name__ == "__main__":
    cli()
