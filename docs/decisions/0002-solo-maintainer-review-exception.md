---
status: accepted
---

# Solo maintainer: no non-author human review

## Context and problem statement

OpenSSF OSPS Baseline QA-07.01, the Scorecard Code-Review check and the Best Practices Badge (gold) expect changes to be approved by someone other than the author. This project has one maintainer.

## Decision outcome

Accept the gap and document it. Mitigations, all enforced on the default branch:

- Every change goes through a pull request; direct pushes and force-pushes are blocked by a ruleset (where the plan allows it).
- The `ci-ok` required check aggregates tests, linters, type checks, SAST, dependency and secret scanning.
- AI review (see `REVIEW.md`) and the `spec-review` skill run on every PR, and review threads must be resolved before merge.
- Dependency updates are held back by a cooldown before they can be merged.

AI review is a mitigation, **not** a substitute for human review; no framework counts it as one.

### Consequences

- Scorecard's Code-Review and Contributors scores stay low; Badge gold is out of reach.
- Revisit when a second maintainer joins: require one approving review in the ruleset.
