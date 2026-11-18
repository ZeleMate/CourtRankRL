# Secrets

Following the OWASP Secrets Management Cheat Sheet: centralise, prefer short-lived credentials, inject at runtime, rotate.

## Where secrets live

| Secret | Store | Used by | Rotation |
|---|---|---|---|
| `SONARQUBE_TOKEN` | Local environment (keychain), never in the repo | `.mcp.json` | yearly |
| `TEMPLATE_READ_TOKEN` (only if the template repo is private) | GitHub Actions secret | `template-update.yml` | fine-grained PAT, contents:read on the template repo, ≤ 1 year |

The machine-readable list is `.github/secret-requirements.json`; the GitHub audit reads it.

Publishing to package registries uses **OIDC trusted publishing**: no stored registry tokens.

## Local development

Pick one and record it here: 1Password CLI (`op run --env-file=…`), Infisical, Doppler, or SOPS + age for encrypted files in the repo. Never put secrets in `.env` files that could be committed, in Docker `ENV`/`ARG`, or in prompts.

## If a secret leaks

1. Revoke/rotate it at the provider first.
2. Check provider logs for use since the leak.
3. Remove it from the code; history rewriting is optional once the secret is dead.
4. Add a note to the changelog's Security section if users are affected.
