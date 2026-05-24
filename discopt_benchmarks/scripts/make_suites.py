#!/usr/bin/env python3
"""Generate stratified instance lists for the small/medium/full benchmark tiers.

Reads ``instancedata.csv`` from the MINLPLib cache and writes three text files
under ``discopt_benchmarks/config/suites/`` (one instance name per line). The
sampling is *stratified* across (category × size) so each tier exercises the
full breadth of MINLPLib, not just easy small instances.

Tiers (target wall time on 8 workers):

  small   ~60 instances, per-instance limit 60s   →  ~10-30 min wall
  medium  ~250 instances, per-instance limit 300s →  ~1.5-3 h wall
  full    every cached instance                    →  ~12-30 h wall on 8 workers

Determinism: stratified sampling is seeded; the same CSV produces the same
lists. Pre-existing lists are overwritten only with ``--force`` when the cache
changes (e.g. MINLPLib refresh).

Usage:
    python -m discopt_benchmarks.scripts.make_suites
    python -m discopt_benchmarks.scripts.make_suites --force
    python -m discopt_benchmarks.scripts.make_suites --tier small
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

# Make sibling utils importable
_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from utils.minlplib_data import InstanceMeta, load_instance_data


SUITES_DIR = _BENCH_ROOT / "config" / "suites"

# (category, size_bucket) → number to draw for the SMALL tier.
# Cells absent from the table get 0 (skipped).
SMALL_PLAN: dict[tuple[str, str], int] = {
    # MINLP — dominant class, sample heavily across sizes
    ("MINLP", "<=10"):   12,
    ("MINLP", "<=30"):   10,
    ("MINLP", "<=100"):   6,
    # Mixed-integer quadratic
    ("MIQP",  "<=30"):    2,
    ("MIQCP", "<=30"):    3,
    ("MIQCP", "<=100"):   3,
    # Continuous quadratic
    ("QP",    "<=30"):    5,
    ("QP",    "<=100"):   5,
    ("QCQP",  "<=30"):    3,
    ("QCQP",  "<=100"):   3,
    # Continuous nonlinear
    ("NLP",   "<=30"):    5,
    ("NLP",   "<=100"):   5,
}  # sum = 62

# MEDIUM tier: broader sampling, including some larger instances.
MEDIUM_PLAN: dict[tuple[str, str], int] = {
    ("MINLP", "<=10"):   30,
    ("MINLP", "<=30"):   30,
    ("MINLP", "<=100"):  30,
    ("MINLP", "<=500"):  20,
    ("MIQP",  "<=30"):    5,
    ("MIQP",  "<=100"):   0,   # only 5 total in this class
    ("MIQCP", "<=30"):    8,
    ("MIQCP", "<=100"):   8,
    ("MIQCP", "<=500"):   4,
    ("QP",    "<=30"):   20,
    ("QP",    "<=100"):  20,
    ("QP",    "<=500"):  15,
    ("QCQP",  "<=30"):    8,
    ("QCQP",  "<=100"):   8,
    ("QCQP",  "<=500"):   5,
    ("NLP",   "<=30"):   15,
    ("NLP",   "<=100"):  15,
    ("NLP",   "<=500"):  10,
}  # sum ≈ 251


def _bucketize(
    index: dict[str, InstanceMeta],
    proven_only: bool,
) -> dict[tuple[str, str], list[str]]:
    """Group instance names by (category, size_bucket)."""
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, meta in index.items():
        if proven_only and not meta.proven_optimal:
            continue
        buckets[(meta.category_bucket, meta.size_bucket)].append(name)
    # Stable order per bucket so the seed produces deterministic output.
    for k in buckets:
        buckets[k].sort()
    return buckets


def _draw_stratified(
    buckets: dict[tuple[str, str], list[str]],
    plan: dict[tuple[str, str], int],
    seed: int,
) -> list[str]:
    """Take min(plan[k], len(bucket[k])) from each cell; report under-fills."""
    rng = random.Random(seed)
    picked: list[str] = []
    for cell, target in sorted(plan.items()):
        pool = buckets.get(cell, [])
        if not pool:
            print(f"  WARN: empty bucket {cell}, wanted {target}", file=sys.stderr)
            continue
        n = min(target, len(pool))
        if n < target:
            print(f"  WARN: bucket {cell} only has {len(pool)}, "
                  f"taking all {n} (wanted {target})", file=sys.stderr)
        picked.extend(rng.sample(pool, n))
    return sorted(set(picked))


def _write_list(path: Path, names: list[str], header_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write(f"# Total instances: {len(names)}\n")
        f.write("#\n")
        for n in names:
            f.write(f"{n}\n")


def make_small(index: dict[str, InstanceMeta], seed: int = 17) -> list[str]:
    buckets = _bucketize(index, proven_only=True)
    return _draw_stratified(buckets, SMALL_PLAN, seed)


def make_medium(index: dict[str, InstanceMeta], seed: int = 17) -> list[str]:
    buckets = _bucketize(index, proven_only=True)
    return _draw_stratified(buckets, MEDIUM_PLAN, seed)


def make_full(index: dict[str, InstanceMeta]) -> list[str]:
    # Full = every instance in the cache (proven or not). We still report
    # outcome buckets, just without an `incorrect` check for the unproven ones.
    return sorted(index.keys())


def _report_distribution(name: str, picks: list[str], index: dict) -> None:
    cells: dict[tuple[str, str], int] = defaultdict(int)
    for p in picks:
        meta = index[p]
        cells[(meta.category_bucket, meta.size_bucket)] += 1
    print(f"\n  {name} distribution ({len(picks)} instances):")
    for (cat, sz), n in sorted(cells.items()):
        print(f"    {cat:6s} {sz:6s}  {n:4d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="MINLPLib cache dir (default: ~/.cache/discopt/minlplib)")
    p.add_argument("--version", type=str, default="current",
                   help="MINLPLib version tag (default: current)")
    p.add_argument("--tier", choices=["small", "medium", "full", "all"], default="all")
    p.add_argument("--seed", type=int, default=17,
                   help="RNG seed for stratified sampling (default: 17)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing list files")
    args = p.parse_args()

    from scripts.fetch_minlplib import get_cache_dir, get_instancedata_path
    cache = args.cache_dir or get_cache_dir()
    csv = get_instancedata_path(cache, args.version)
    if not csv.exists():
        print(f"ERROR: instancedata.csv missing at {csv}\n"
              "Run: python -m discopt_benchmarks.scripts.fetch_minlplib --skip-archive",
              file=sys.stderr)
        sys.exit(1)

    print(f"[index] loading {csv}")
    index = load_instance_data(csv)
    n_proven = sum(1 for m in index.values() if m.proven_optimal)
    print(f"[index] {len(index)} instances ({n_proven} proven-optimal)")

    targets = []
    if args.tier in {"small", "all"}:
        targets.append(("small", make_small(index, seed=args.seed),
                        [f"discopt small tier — stratified MINLPLib sample (seed={args.seed})",
                         "Target wall time ~30 min on 8 workers at 60s per-instance limit",
                         "Source: instancedata.csv (proven-optimal only)"]))
    if args.tier in {"medium", "all"}:
        targets.append(("medium", make_medium(index, seed=args.seed),
                        [f"discopt medium tier — stratified MINLPLib sample (seed={args.seed})",
                         "Target wall time ~2 h on 8 workers at 300s per-instance limit",
                         "Source: instancedata.csv (proven-optimal only)"]))
    if args.tier in {"full", "all"}:
        targets.append(("full", make_full(index),
                        ["discopt full tier — every MINLPLib instance",
                         "Target wall time 12-30 h on 8 workers at 600s per-instance limit",
                         "Source: instancedata.csv (proven + unproven)"]))

    for name, picks, hdr in targets:
        out = SUITES_DIR / f"{name}.txt"
        if out.exists() and not args.force:
            print(f"[skip] {out} exists (use --force to overwrite)")
            _report_distribution(name, picks, index)
            continue
        _write_list(out, picks, hdr)
        print(f"[wrote] {out} ({len(picks)} instances)")
        _report_distribution(name, picks, index)


if __name__ == "__main__":
    main()
