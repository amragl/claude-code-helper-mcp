"""Agent Forge hook integration for automatic memory recording during pipeline execution.

This package provides hook functions that integrate with Agent Forge's pipeline
lifecycle events.  When registered, these hooks transparently record memory
entries (steps, files, branches, decisions) as the pipeline executes, without
requiring explicit MCP tool calls from the agents.

Hook functions:

- :func:`post_tool_call` -- Records a step after each significant tool invocation.
- :func:`post_build_start` -- Records the start of a build phase (branch creation,
  task start).
- :func:`post_build_complete` -- Records build completion (PR creation, file summary).
- :func:`post_merge` -- Records the merge event and completes the memory task.

State readers:

- :class:`StateReader` -- Reads Agent Forge state files (pipeline.json,
  build-output.json, backlog.json, plan-output.json) with caching and graceful
  error handling.

All hooks are designed to be idempotent and failure-tolerant: if the memory
system is unavailable, hooks log a warning and return without raising exceptions.
"""

from claude_code_helper_mcp.hooks.pipeline import (
    post_build_complete,
    post_build_start,
    post_merge,
    post_tool_call,
)
from claude_code_helper_mcp.hooks.state_reader import (
    BacklogPhase,
    BacklogState,
    BacklogTicket,
    BuildOutput,
    PipelineState,
    PlanOutput,
    StateReader,
    TicketContext,
)

__all__ = [
    "post_tool_call",
    "post_build_start",
    "post_build_complete",
    "post_merge",
    "BacklogPhase",
    "BacklogState",
    "BacklogTicket",
    "BuildOutput",
    "PipelineState",
    "PlanOutput",
    "StateReader",
    "TicketContext",
]
