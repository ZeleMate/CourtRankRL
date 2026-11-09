# Security policy

CourtRankRL is a thesis research project: retrieval and reranking of Hungarian court decisions. It is a personal project with one maintainer ([@ZeleMate](https://github.com/ZeleMate)). Only the latest commit on `main` is supported; there are no releases.

## Reporting a vulnerability

Please do not open a public issue, pull request or discussion. Report it privately through GitHub: **Security → Report a vulnerability** on this repository.

Include what you found, how to reproduce it, and the impact you expect. You will get an acknowledgement within 14 days, and confirmed issues are fixed or documented with a mitigation. Credit goes in the fix's commit unless you prefer otherwise. This is a hobby project, so there is no bounty.

## Scope

In scope: the code in `src/`, `scripts/` and `configs/`, the notebooks and the project configuration.

Because this project sends retrieved documents and user input to an LLM, these also count as vulnerabilities even when the root cause is model behaviour rather than a code bug:

- prompt injection through documents, retrieved chunks or user input that makes the system take actions or reveal data it otherwise would not;
- API keys or other secrets leaking into logs, notebook outputs, traces or model prompts.

Out of scope: weaknesses in third-party model providers or dependencies themselves (report those upstream), and anything that needs an attacker who can already edit your local `.env` or source code.
