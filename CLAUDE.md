# CLAUDE.md

Project guidance for Claude Code (claude.ai/code).

The canonical agent-facing reference is **[`AGENTS.md`](AGENTS.md)** at
the repo root. Read it first. It documents hardware constraints,
toolchain, build entry points, code conventions, correctness
tolerances, the SASS hand-edit workflow, and the four laws of GA104.

Everything below is Claude Code-specific and additive to `AGENTS.md`.

## Claude Code addenda

### Team activation

When the user asks to activate or use a team:

1. Call `ToolSearch("select:TeamCreate")` to load the TeamCreate tool.
2. Read the team definition from
   `/mnt/d/dev/p/agent-almanac/teams/<team-name>.md`.
3. Call `TeamCreate` with the team configuration.

Do not fall back to spawning individual agents via the Agent tool —
always use TeamCreate for team requests. Available teams are listed
in `/mnt/d/dev/p/agent-almanac/teams/_registry.yml`.

### Skills

When the user invokes a skill (`/breathe`, `/meditate`, `/heal`,
`/transmute`, `/athanor`, `/rest`), read the skill's `SKILL.md` from
`/mnt/d/dev/p/agent-almanac/skills/<name>/SKILL.md` and follow the
procedure documented there. The skills are AI self-regulation
patterns, not project commands.

### Session handoff

`docs/CONTINUE_HERE.md` is a session-scoped scratchpad. At session
close, update it with the work completed this session and the next
concrete steps. Treat it as the rolling handoff document, not as
durable documentation.
