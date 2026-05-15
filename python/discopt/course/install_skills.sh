#!/usr/bin/env bash
# Install the discopt course's slash commands and assessor skill into the
# project-level .claude/ directory.
#
# Usage:
#   bash course/install_skills.sh           # install into ./.claude (project)
#   bash course/install_skills.sh --user    # install into ~/.claude (global)
#
# Idempotent: safely re-runnable. Existing files are overwritten only with
# --force.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/_claude_assets"

DEST=".claude"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --user)  DEST="$HOME/.claude" ;;
    --force) FORCE=1 ;;
    *)       echo "unknown flag: $arg"; exit 1 ;;
  esac
done

mkdir -p "$DEST/commands/course" "$DEST/skills/course-assessor"

copy() {
  local src="$1" dst="$2"
  if [[ -e "$dst" && $FORCE -eq 0 ]]; then
    echo "  skip (exists): $dst"
  else
    cp "$src" "$dst"
    echo "  ok           : $dst"
  fi
}

echo "Installing course commands -> $DEST/commands/course/"
for f in "$SRC"/commands/course/*.md; do
  copy "$f" "$DEST/commands/course/$(basename "$f")"
done

echo "Installing course-assessor skill -> $DEST/skills/course-assessor/"
copy "$SRC/skills/course-assessor/SKILL.md" "$DEST/skills/course-assessor/SKILL.md"

echo "Done. Slash commands available: /course:lesson, /course:assess, /course:hint, /course:progress, /course:grade-writing, /course:cite-check"
