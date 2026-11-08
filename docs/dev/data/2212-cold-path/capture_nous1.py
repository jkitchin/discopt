#!/usr/bin/env python3
"""#2212: re-capture performance-plan §121's undecided nous1 node LPs.

§121's capture: nous1 (`python/tests/data/issue_1871/nous1.nl`) with the native
kernel's fixed rows permuted by seed 5 (`2202-node-lp-slack-retry/perm_probe.py`),
`DISCOPT_NODE_LP_SLACK_RETRY=0`, 60 s, every node LP recorded through the kernel's
`node_observer`; the undecided ones (`numerical` / `iter_limit`) are what #2212's
"done" list asks the engine to decide (>= 36 of 40). This captures the same way,
with `DISCOPT_LP_LOGICAL_DUAL_COLD` unset (the defect as callers see it today), and
writes the first `--keep` undecided LPs, with the basis each started from, to an
`.npz` for `replay_nous1.py`.

    DISCOPT_NODE_LP_SLACK_RETRY=0 python -u capture_nous1.py --out nous1_undecided.npz

Prints the totals and an executed count; exits non-zero when nothing undecided was
captured (CLAUDE.md §6). #2202 found the defect's onset differs between macOS and
Linux (row order changes the floating-point work), so the count here need not
match §121's 590.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "python"))
import discopt  # noqa: E402
import discopt._rust as R  # noqa: E402
import discopt.modeling as dm  # noqa: E402

assert Path(discopt.__file__).resolve().is_relative_to(REPO), discopt.__file__
assert os.environ.get("DISCOPT_LP_LOGICAL_DUAL_COLD", "0") in ("", "0"), "capture runs flag OFF"
assert os.environ.get("DISCOPT_NODE_LP_SLACK_RETRY") == "0", "capture runs the retry OFF"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--time-limit", type=float, default=60.0)
ap.add_argument("--keep", type=int, default=40)
ap.add_argument("--out", required=True)
args = ap.parse_args()

orig = R.solve_spatial_tree_py
seen = {"nodes": 0, "undecided": 0, "warm_undecided": 0}
kept: list[dict] = []


def observe(rec):
    seen["nodes"] += 1
    if rec["status"] not in ("numerical", "iter_limit"):
        return
    seen["undecided"] += 1
    if "start_col_status" not in rec:
        return
    seen["warm_undecided"] += 1
    if len(kept) < args.keep:
        kept.append(
            {k: (np.asarray(v).copy() if hasattr(v, "__len__") else v) for k, v in rec.items()}
        )


def wrap(*a, **k):
    if args.seed and int(k.get("max_nodes", 10**9)) > 1:  # perm_probe.py's permutation
        ptr = np.asarray(k["fixed_row_ptr"])
        cols = np.asarray(k["fixed_cols"])
        co = np.asarray(k["fixed_coeffs"])
        rhs = np.asarray(k["fixed_rhs"])
        perm = np.random.default_rng(args.seed).permutation(len(rhs))
        nc, ncf, np_ = [], [], [0]
        for q in perm:
            nc.append(cols[ptr[q] : ptr[q + 1]])
            ncf.append(co[ptr[q] : ptr[q + 1]])
            np_.append(np_[-1] + ptr[q + 1] - ptr[q])
        k["fixed_row_ptr"] = np.array(np_, dtype=ptr.dtype)
        k["fixed_cols"] = np.concatenate(nc).astype(cols.dtype)
        k["fixed_coeffs"] = np.concatenate(ncf)
        k["fixed_rhs"] = rhs[perm].copy()
    k["node_observer"] = observe
    return orig(*a, **k)


R.solve_spatial_tree_py = wrap
r = dm.from_nl(str(REPO / "python/tests/data/issue_1871/nous1.nl")).solve(
    time_limit=args.time_limit, gap_tolerance=1e-4
)
print(f"nous1 seed={args.seed}: {r.status} nodes={r.node_count} bound={r.bound}", flush=True)
print(
    f"node LPs observed {seen['nodes']}, undecided {seen['undecided']}, "
    f"warm-started undecided {seen['warm_undecided']}, kept {len(kept)}",
    flush=True,
)
flat = {}
for i, rec in enumerate(kept):
    for key in (
        "m",
        "n",
        "col_ptr",
        "row_idx",
        "vals",
        "b",
        "l",
        "u",
        "c",
        "start_col_status",
        "start_basic_vars",
        "status",
    ):
        flat[f"{i}_{key}"] = rec[key]
np.savez_compressed(args.out, count=len(kept), **flat)
print(f"EXECUTED: captured {len(kept)} undecided node LPs -> {args.out}", flush=True)
sys.exit(0 if kept else 1)
