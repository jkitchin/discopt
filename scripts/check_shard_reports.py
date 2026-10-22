#!/usr/bin/env python
"""Fail unless the CI shards' ``--shard-report`` files add up to the whole lane.

Each shard writes {shard_id, num_shards, total, mine, expected_s}. The lane ran
completely only if every shard reported, all agree on ``total``, and the
``mine`` counts sum to it (python/tests/_sharding.py). A missing report means a
shard died before collection; that is a failure, not a smaller lane.

    python scripts/check_shard_reports.py --num-shards 4 reports/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("reports", nargs="*", type=Path)
    args = ap.parse_args()

    reps = [json.loads(p.read_text()) for p in args.reports]
    for r in sorted(reps, key=lambda r: r["shard_id"]):
        print(
            f"shard {r['shard_id']}/{r['num_shards']}: {r['mine']} of {r['total']} tests, "
            f"expected {r['expected_s']} s"
        )
    errors = []
    ids = sorted(r["shard_id"] for r in reps)
    if ids != list(range(args.num_shards)):
        errors.append(f"expected reports from shards 0..{args.num_shards - 1}, got {ids}")
    if {r["num_shards"] for r in reps} != {args.num_shards}:
        errors.append(f"shards disagree on num_shards: {[r['num_shards'] for r in reps]}")
    totals = {r["total"] for r in reps}
    if len(totals) != 1:
        errors.append(f"shards collected different selections: totals {sorted(totals)}")
    elif (total := totals.pop()) <= 0:
        errors.append("the lane selected zero tests")
    elif (ran := sum(r["mine"] for r in reps)) != total:
        errors.append(f"shards ran {ran} tests, the selection has {total}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    if not errors:
        print(f"OK: {len(reps)} shards cover all {sum(r['mine'] for r in reps)} selected tests")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
