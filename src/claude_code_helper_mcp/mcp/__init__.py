"""FastMCP server and tool definitions for memory recording and retrieval."""


def create_server():
    """Factory function for creating the MCP server instance.

    This is the entry point registered in pyproject.toml under
    [project.entry-points."mcp.servers"]. The actual server setup
    will be implemented in CMH-006 (FastMCP server bootstrap).
    """
    raise NotImplementedError(
        "MCP server not yet implemented. See CMH-006: FastMCP server bootstrap."
    )
