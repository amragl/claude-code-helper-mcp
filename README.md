# claude-code-helper-mcp

Structured memory system for Claude Code sessions. Provides an MCP server and CLI for recording task context, decisions, file changes, and branch operations during development -- enabling recovery after `/clear` and drift detection across sessions.

## Features

- **Task Memory** -- Record steps, decisions, file modifications, and branch operations per task
- **Sliding Window** -- Retain context for the current task plus 3 recent tasks
- **MCP Server** -- FastMCP-based tools for transparent memory recording
- **Recovery Context** -- Full task context restoration after `/clear`
- **Drift Detection** -- Identify error loops, scope creep, and plan drift
- **CLI** -- Developer-facing commands for memory inspection and management
- **Agent Forge Integration** -- Automatic recording via pipeline hooks

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Show version
memory --version

# Show help
memory --help
```

### MCP Server

The MCP server is registered as `claude-code-helper` and communicates over stdio transport. Configure it in your Claude Code MCP settings.

## Project Structure

```
src/claude_code_helper_mcp/
    __init__.py          # Package init with version
    models/              # Pydantic data models
    storage/             # File-based storage engine
    mcp/                 # FastMCP server and tools
    cli/                 # Click CLI commands
    detection/           # Drift and confusion detection
    hooks/               # Agent Forge hook integration
tests/                   # Test suite
```

## Development

```bash
# Run tests
pytest tests/

# Run CLI directly
python -m claude_code_helper_mcp.cli.main --help
```

## Storage

Task memory is persisted to `.claude-memory/` directories within each project. Each task gets its own JSON file containing steps, decisions, file records, and branch records.

## License

MIT
