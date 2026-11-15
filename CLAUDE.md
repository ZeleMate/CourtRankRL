@AGENTS.md

## Claude Code specifics

- Before opening a PR, run the `spec-review` skill (diff vs. the linked issue/ADR) and `/code-review`.
- Hard limits are enforced in `.claude/settings.json` (deny rules) and hooks, not in this file. A PostToolUse hook formats files you edit.
- Path-specific rules live in `.claude/rules/`.
- Put personal preferences in `CLAUDE.local.md` only if you also keep the `@AGENTS.md` import here; Claude Code otherwise skips AGENTS.md.
