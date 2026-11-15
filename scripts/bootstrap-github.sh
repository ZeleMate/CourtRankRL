#!/usr/bin/env bash
# Create the GitHub repo (if needed) and apply the template's settings. Idempotent: safe to re-run.
#
#   ./scripts/bootstrap-github.sh --dry-run   # print what would happen
#   ./scripts/bootstrap-github.sh             # do it
#
# Needs: gh (authenticated as ZeleMate or an org admin), git.
set -euo pipefail

OWNER="ZeleMate"
REPO="CourtRankRL"
VISIBILITY="public"
PLAN="free"
PUBLISHES="no"
DESCRIPTION="Retrieval and GRPO reranking over Hungarian court decisions"
FULL="$OWNER/$REPO"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/.."

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }

# Run a command, or print it in dry-run mode. Failures are reported, not fatal,
# because some endpoints depend on plan or visibility.
run() {
  if [ "$DRY_RUN" = 1 ]; then printf '  [dry-run] %s\n' "$*"; return 0; fi
  if out=$("$@" 2>&1); then ok "$*"; else warn "failed: $*"; printf '    %s\n' "$out" | head -5; fi
}

api() { run gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" "$@"; }

if [ "$DRY_RUN" = 0 ]; then
  gh auth status >/dev/null 2>&1 || { echo "Run 'gh auth login' first."; exit 1; }
fi

# ------------------------------------------------------------------ git + repo
say "Local git repository"
if [ ! -d .git ]; then
  run git init -b main
fi
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  run git add -A
  run git commit -m "chore: initial commit from repo template"
else
  ok "already has commits"
fi

say "GitHub repository $FULL ($VISIBILITY)"
if gh repo view "$FULL" >/dev/null 2>&1; then
  ok "exists"
else
  run gh repo create "$FULL" "--$VISIBILITY" --description "$DESCRIPTION" --source . --remote origin --push
fi

# ------------------------------------------------------------------ repo settings
say "Merge settings: squash only, PR title as commit, delete merged branches"
api -X PATCH "repos/$FULL" \
  -F allow_squash_merge=true -F allow_merge_commit=false -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true -F allow_auto_merge=true -F has_wiki=false \
  -f squash_merge_commit_title=PR_TITLE -f squash_merge_commit_message=PR_BODY

say "Actions: read-only default token, no PR approvals by Actions, SHA pinning required"
api -X PUT "repos/$FULL/actions/permissions/workflow" \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false
api -X PUT "repos/$FULL/actions/permissions" -F enabled=true -f allowed_actions=all -F sha_pinning_required=true

say "Dependabot alerts and security updates"
api -X PUT "repos/$FULL/vulnerability-alerts"
api -X PUT "repos/$FULL/automated-security-fixes"

if [ "$VISIBILITY" = public ]; then
  say "Public-repo security features (free)"
  api -X PATCH "repos/$FULL" --input - <<'JSON'
{"security_and_analysis": {
  "secret_scanning": {"status": "enabled"},
  "secret_scanning_push_protection": {"status": "enabled"}
}}
JSON
  api -X PUT "repos/$FULL/private-vulnerability-reporting"
  api -X PATCH "repos/$FULL/code-scanning/default-setup" -f state=configured -f query_suite=default
  say "Fork PRs: workflows from every outside contributor wait for approval"
  api -X PUT "repos/$FULL/actions/permissions/fork-pr-contributor-approval" -f approval_policy=all_external_contributors
else
  say "Private repo: GitHub code/secret scanning need a paid org add-on; Semgrep, OSV-Scanner and Gitleaks run in CI instead"
fi

# ------------------------------------------------------------------ ruleset
say "Default-branch ruleset (PR required, ci-ok required, no force-push/delete, linear history)"
if [ "$VISIBILITY" = public ] || [ "$PLAN" = pro ]; then
  existing=""
  if [ "$DRY_RUN" = 0 ]; then
    existing=$(gh api "repos/$FULL/rulesets" --jq '.[] | select(.name=="main") | .id' 2>/dev/null || true)
  fi
  if [ -n "$existing" ]; then
    api -X PUT "repos/$FULL/rulesets/$existing" --input .github/rulesets/main.json
  else
    api -X POST "repos/$FULL/rulesets" --input .github/rulesets/main.json
  fi
else
  warn "Skipped: rulesets on private repos need GitHub Pro. The gap is recorded in docs/THREAT_MODEL.md."
fi

# ------------------------------------------------------------------ releases
if [ "$PUBLISHES" = yes ]; then
  say "Release settings: immutable releases, 'release' environment limited to main"
  api -X PUT "repos/$FULL/immutable-releases"
  if [ "$VISIBILITY" = public ] || [ "$PLAN" = pro ]; then
    api -X PUT "repos/$FULL/environments/release" --input - <<'JSON'
{"deployment_branch_policy": {"protected_branches": false, "custom_branch_policies": true}}
JSON
    api -X POST "repos/$FULL/environments/release/deployment-branch-policies" -f name=main -f type=branch
  else
    warn "Skipped environment: environments on private repos need GitHub Pro."
  fi
fi

# ------------------------------------------------------------------ manual steps
say "Done. Remaining manual steps (also in docs/SETUP.md):"
sed -n '/^## Manual steps/,$p' docs/SETUP.md | tail -n +2
