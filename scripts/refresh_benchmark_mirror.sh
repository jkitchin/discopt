#!/usr/bin/env bash
# Refresh the local MINLPLib benchmark mirror used by every measurement harness.
#
# WHY: the canonical snapshot lives in Dropbox (deliberately — it is the backed-up
# copy), but reading a 926 MB corpus through a sync folder makes every timing panel a
# measurement of the sync daemon as well as the solver. Dropbox's indexer wakes on its
# own schedule; observed at 121% CPU during a timing run on 2026-07-27. The mirror
# removes that dependency rather than trying to predict it.
#
# The mirror is READ-ONLY reference data. Nothing writes here: solver harnesses stage
# into tempfile.mkdtemp (verified in global_opt_baron_vs_discopt.py:442). If you ever
# see a .sol appear in the mirror, a harness is running in-place and should be fixed.
#
# Usage:
#   scripts/refresh_benchmark_mirror.sh            # sync new/changed files
#   scripts/refresh_benchmark_mirror.sh --check    # report drift, change nothing
set -uo pipefail

SRC="${DISCOPT_MINLP_BENCH_SRC:-$HOME/Dropbox/projects/discopt-minlp-benchmark}"
DST="${DISCOPT_MINLP_BENCH:-$HOME/projects/discopt-minlp-benchmark}"

if [ ! -f "$SRC/minlplib.solu" ]; then
  echo "FATAL: canonical snapshot not found at $SRC (no minlplib.solu)." >&2
  echo "       Set DISCOPT_MINLP_BENCH_SRC to the snapshot root." >&2
  exit 2
fi

# macOS ships rsync 2.6.9 — no --info=, no --outbuf. Keep the flags portable; an
# unsupported flag makes rsync print usage and exit 0-ish, which looks like success
# and silently leaves an EMPTY mirror. (That happened once; hence the verify below.)
FLAGS=(-a --stats --exclude 'papers/' --exclude '__pycache__/' --exclude '*.sol')

if [ "${1:-}" = "--check" ]; then
  echo "# dry run: $SRC -> $DST"
  rsync "${FLAGS[@]}" --dry-run "$SRC/" "$DST/" | tail -12
  exit 0
fi

mkdir -p "$DST"
echo "# syncing $SRC -> $DST (excluding papers/, __pycache__, *.sol)"
rsync "${FLAGS[@]}" "$SRC/" "$DST/" | tail -6

# VERIFY, do not assume. An empty or partial mirror shadows the canonical snapshot in
# corpus.py's resolution order, which would make every oracle check vacuous while
# looking clean.
src_nl=$(find "$SRC/minlplib/nl" -name '*.nl' 2>/dev/null | wc -l | tr -d ' ')
dst_nl=$(find "$DST/minlplib/nl" -name '*.nl' 2>/dev/null | wc -l | tr -d ' ')
src_opt=$(grep -c '=opt=' "$SRC/minlplib.solu" 2>/dev/null || echo 0)
dst_opt=$(grep -c '=opt=' "$DST/minlplib.solu" 2>/dev/null || echo 0)

echo "verify: .nl  $dst_nl / $src_nl"
echo "verify: =opt= $dst_opt / $src_opt"

if [ "$dst_nl" != "$src_nl" ] || [ "$dst_opt" != "$src_opt" ] || [ "$dst_nl" = "0" ]; then
  echo "FATAL: mirror is incomplete — refusing to leave it in place as a shadow." >&2
  exit 1
fi

case "$DST" in
  *Dropbox*|*"Google Drive"*|*OneDrive*)
    echo "WARNING: the mirror itself is inside a sync folder ($DST) — that defeats the point." >&2
    ;;
esac

echo "OK: mirror complete at $DST ($(du -sh "$DST" | cut -f1)). Harnesses resolve it"
echo "    automatically via discopt_benchmarks.utils.corpus.corpus_root()."
