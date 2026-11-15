# Review rubric

Used by AI reviewers (the Claude review workflow, the `code-reviewer` subagent) and by you when self-reviewing. Formatting, lint and types are already enforced by CI; do not comment on them.

Report only findings you can point to in the diff. Prefer a concrete suggested change over a description. Say "no findings" when there are none.

## 1. Does it do what was asked?

- Find the linked issue, ADR or spec (PR body, branch name, commit message). List each requirement and whether the diff meets it.
- Flag scope creep: changes not asked for.

## 2. Correctness

- Logic errors, off-by-one, wrong conditions, unhandled `None`/`undefined`/`Err`.
- Error handling: failures surfaced, not swallowed; retries bounded.
- Concurrency and async: missing `await`, races, shared mutable state.
- Tests: do they test the behaviour, including failure paths, or just the happy path?

## 3. Security

- Secrets, tokens or personal data in code, logs, errors or test fixtures.
- Injection: user or model output reaching shell, SQL, HTML, file paths, or `eval`.
- AuthN/AuthZ checks on every new endpoint or tool.
- New dependencies: real, maintained, needed, licence-compatible.
- CI changes: permissions widened, actions not SHA-pinned, untrusted input in `run:`.

## Severity

- **blocker**: wrong behaviour, security issue, data loss.
- **should-fix**: missing tests for changed behaviour, unclear error handling.
- **nit**: only if it changes readability materially. Keep nits to a minimum.
