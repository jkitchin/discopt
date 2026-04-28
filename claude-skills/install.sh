#!/usr/bin/env bash
# Legacy wrapper around `discopt install-skills`.
#
# Skills and agents now live inside the Python package at
# python/discopt/skills/ and ship with the wheel, so downstream users
# can install them via `discopt install-skills`. This script forwards
# to that CLI with --project-scope so existing users of
# `bash claude-skills/install.sh` keep getting the project-local
# behavior they had before.
#
# Usage:
#   bash claude-skills/install.sh           # copy into ./.claude/
#   bash claude-skills/install.sh --dev     # symlink instead of copy
#   bash claude-skills/install.sh --force   # overwrite existing

set -euo pipefail

exec discopt install-skills --project-scope "$@"
