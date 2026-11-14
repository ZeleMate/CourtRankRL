#!/usr/bin/env bash
# PostToolUse hook: format the file Claude just edited. Never blocks (always exits 0).
set -uo pipefail

command -v jq >/dev/null 2>&1 || exit 0
file=$(jq -r '.tool_input.file_path // empty')
[ -n "$file" ] && [ -f "$file" ] || exit 0
root="${CLAUDE_PROJECT_DIR:-.}"

case "$file" in
  # No managed language: add a pattern and your own formatter here, for example
  #   *.py) ruff format --quiet "$file" >/dev/null 2>&1 ;;
esac
exit 0
