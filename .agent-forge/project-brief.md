# Project Brief: claude-memory-helper

## Overview

A companion MCP server and CLI agent for Agent Forge that actively records structured memory during task execution, enabling seamless context recovery after `/clear`. It also monitors Claude Code's behavior for context drift, error loops, and confusion — intervening with graduated warnings and automatic context resets to keep the pipeline on track.

## Objectives

1. Actively record task context (files touched, branches, steps taken, decisions made) during Agent Forge pipeline execution
2. Maintain a sliding window of memory: last 3 completed tasks + current task
3. Enable seamless `/clear` recovery — Agent Forge reads helper memory to resume with full awareness, eliminating lossy context compaction
4. Integrate into the Agent Forge pipeline via hooks for automatic recording without manual intervention
5. Detect context drift, error loops, confusion patterns, and scope creep by monitoring Claude Code actions against the recorded plan
6. Implement graduated intervention — warn on first detection, auto `/clear` + reload on second detection
7. Provide a developer-facing CLI for reviewing task history, memory state, and intervention logs

## Target Users

- **Agent Forge agents** — primary consumer; machine-to-machine integration via MCP tools and hooks
- **Developers** — secondary; CLI for reviewing what happened, debugging pipeline issues, and understanding task history

## Tech Stack

- **Languages:** Python
- **Frameworks:** Click (CLI), FastMCP (MCP server)
- **Databases:** None (file-based storage only)
- **APIs/Services:** Claude Code MCP protocol, Agent Forge state files and hooks
- **Infrastructure:** Local file system (JSON + markdown memory files)

## Requirements

### Must Have (P0)

1. MCP server exposing memory management tools (`record_step`, `record_file`, `record_branch`, `record_decision`, `get_recovery_context`, `check_alignment`)
2. Structured memory storage with sliding window retention (3 completed tasks + current)
3. Recovery context tool that provides full task context after `/clear` — ticket info, files modified, branches, last steps completed, next steps planned
4. Click CLI for viewing memory state (`memory status`, `memory show <task>`, `memory list`)
5. Agent Forge hook integration — post-tool-call hooks that automatically record actions without requiring explicit tool calls from the agent
6. Memory file format: structured JSON for machine consumption + markdown summaries for human readability

### Should Have (P1)

1. Plan drift detection — compare current Claude Code actions (file edits, bash commands) against the expected plan from memory
2. Error loop detection — track consecutive failures of the same action and trigger intervention after threshold (e.g., 3 consecutive failures)
3. Confusion pattern detection — flag when Claude references non-existent files, wrong function names, or contradicts recorded state
4. Scope creep detection — flag when Claude edits files or adds features outside the current ticket scope
5. Graduated intervention system — first detection triggers a warning injected via hook; second detection triggers auto `/clear` + memory reload suggestion

### Nice to Have (P2)

1. Memory analytics — patterns across many tasks (common error types, average steps per ticket, frequently modified files)
2. Developer review dashboard — formatted CLI output showing task timelines and decision trees
3. Memory export/import — share context snapshots between sessions or developers
4. Cross-project memory — track patterns across Agent Forge projects, not just within one

## Constraints

- Must be lightweight — near-zero overhead on pipeline execution; recording should not noticeably slow down Agent Forge
- File-based storage only — no external databases; structured JSON/markdown files for full portability
- Must work as both MCP server (for agent integration) and CLI (for developer review)
- Memory files should be human-readable when inspected directly
- Sliding window must hard-cap at 3+current to prevent unbounded growth

## Existing Codebase

- **Starting from scratch:** yes
- **Existing repo:** https://github.com/amragl/claude-memory-helper
- **Current state:** empty (just initialized with Agent Forge)
- **Technical debt:** none

## Dependencies

- **Claude Code** — MCP server protocol (FastMCP SDK)
- **Agent Forge** — hooks system, state files (pipeline.json, backlog.json), project config
- **Click** — CLI framework for developer-facing commands
- **Agent Forge state files** — reads pipeline.json, backlog.json, build-output.json to understand current task context

## Success Criteria

1. Memory recording works automatically during Agent Forge pipeline execution via hooks — no manual tool calls needed
2. After `/clear`, Agent Forge reads recovery context and resumes the next ticket with full awareness of previous work
3. Sliding window correctly maintains exactly 3 completed tasks + current, rotating oldest out
4. Drift detection catches basic error loops (3+ consecutive same-action failures) and wrong-file edits (files outside ticket scope)
5. Graduated intervention functions: warning on first drift, auto-clear suggestion on second
6. Developer CLI (`memory status`, `memory show`, `memory list`) displays clear, readable task history
7. Full pipeline integration adds less than 500ms overhead per ticket
8. Memory files are valid JSON and human-readable markdown

## Notes

- The core value proposition is replacing lossy context compaction with deliberate, structured memory. Agent Forge currently loses important details when context is compressed — this project eliminates that problem entirely.
- The MCP server should expose tools that are intuitive for Claude Code to call: simple names, clear descriptions, minimal required parameters.
- Hook integration is critical — the recording should happen automatically as a side effect of Agent Forge working, not require the agent to explicitly "remember" things.
- The drift detection feature is what transforms this from a simple logger into an active quality guardian. It should be designed to minimize false positives — only intervene when there's clear evidence of confusion, not just unexpected behavior.
- Consider using Agent Forge's existing `.agent-forge/state/` directory patterns for consistency, but store memory files in a separate `.claude-memory/` directory to keep concerns separated.
