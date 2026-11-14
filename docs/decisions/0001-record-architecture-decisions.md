---
status: accepted
---

# Record architecture decisions

## Context and problem statement

Decisions about architecture, dependencies and security trade-offs get lost in chat logs and commit messages. Coding agents and future maintainers need a durable record of what was decided and why.

## Decision outcome

Use [MADR 4.0](https://adr.github.io/madr/) files in `docs/decisions/`, named `NNNN-short-title.md`, one decision per file. Superseded decisions stay in place with `status: superseded by NNNN`.

Write an ADR when a change:

- adds or replaces a framework, datastore, model provider or external service;
- changes a security boundary (auth, secrets, agent tool permissions);
- accepts a known risk or deviates from the repo template.

### Consequences

- Good: agents and reviewers can check a change against recorded decisions (`spec-review` skill).
- Bad: a few minutes of writing per significant decision.
