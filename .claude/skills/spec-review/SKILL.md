---
name: spec-review
description: Check the current branch's diff against the issue, ADR or spec it is meant to implement, and report unmet requirements and scope creep. Use before opening a PR, or when asked whether a change "does what was asked".
---

# Spec review

Goal: say whether this diff implements what was asked, and nothing else. Correctness and style are covered by `/code-review` and CI; stay on requirements.

1. **Find the spec.** In order: the PR body (`gh pr view --json body,title`), an issue number in the branch name or commits (`gh issue view <n>`), ADRs touched or referenced under `docs/decisions/`, or ask the user. If none exists, stop and say so.
2. **List the requirements** as short, checkable statements. Include acceptance criteria and explicit non-goals.
3. **Read the diff**: `git diff --merge-base origin/main`. Read surrounding code where needed.
4. **Check each requirement**: met / partly met / not met / can't tell, with the file and line that shows it.
5. **Check tests**: is each requirement covered by a test that would fail without the change?
6. **Flag scope creep**: changed behaviour no requirement asked for.
7. **Report** in this shape:

```
Spec: <link or path>
| Requirement | Status | Evidence |
|---|---|---|
Untested requirements: …
Scope creep: …
Verdict: ready / needs work
```

Do not edit code during the review.
