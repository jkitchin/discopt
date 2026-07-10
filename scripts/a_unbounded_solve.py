#!/usr/bin/env python
"""A-UNBOUNDED (task #94): solve one instance and report the bound trajectory.

Usage: python scripts/a_unbounded_solve.py <instance> [time_limit_s]

Set DISCOPT_ROOT_FIXPOINT=1 DISCOPT_NODE_REDUCE=1 to reproduce the Lever-2
inertness result (docs/dev/a-unbounded-entry-2026-07-10.md section 3): on
nvs05/tanksize the flags leave root_bound, final bound, and node count unchanged
because both hang off ``_mc_lp_relaxer is not None`` and this class routes to
alphaBB/interval per-node.
"""

from __future__ import annotations

import os
import sys

import discopt.modeling as dm

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "nvs05"
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    m = dm.from_nl(NL + name + ".nl")
    res = m.solve(time_limit=tl)
    print(
        f"{name}: status={res.status} obj={res.objective:.4f} "
        f"bound={res.bound:.4f} root_bound={res.root_bound} "
        f"root_gap={res.root_gap} gap={res.gap:.3f} "
        f"cert={res.gap_certified} nodes={res.node_count}"
    )


if __name__ == "__main__":
    main()
