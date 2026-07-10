#!/usr/bin/env python
"""A-UNBOUNDED (task #94): McCormick-LP root probe, raw box vs FBBT-tightened box.

Shows the wiring the entry experiment isolated (docs/dev/a-unbounded-entry-2026-07-10.md
sections 4-5): on the RAW unbounded box the McCormick-LP root probe returns
``numerical`` (tanksize) or ``optimal``-with-no-safe-bound (casctanks), which nulls the
LP relaxer (solver.py:5176) and disables OBBT/node-reduce. FBBT-finitizing the box first
flips the probe to a usable bound -- engine robustness, NOT a certification lever (see
the doc's section 3 end-to-end table: no gap closed).
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import numpy as np
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt.tightening import fbbt_box

NL = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
INSTS = ["nvs05", "tanksize", "casctanks"]


def apply_fbbt(m):
    r = fbbt_box(m, max_iter=50)
    k = 0
    for v in m._variables:
        sz = v.size
        v.lb = np.asarray(r.lb[k : k + sz], float).reshape(v.shape)
        v.ub = np.asarray(r.ub[k : k + sz], float).reshape(v.shape)
        k += sz
    return np.asarray(r.lb, float), np.asarray(r.ub, float)


def probe(name: str, use_fbbt: bool) -> None:
    m = dm.from_nl(NL + name + ".nl")
    if use_fbbt:
        lb, ub = apply_fbbt(m)
    else:
        lb, ub = flat_variable_bounds(m)
    rel = MccormickLPRelaxer(m)
    p = rel.solve_at_node(lb, ub, time_limit=5.0)
    useful = p is not None and (p.status == "infeasible" or p.lower_bound is not None)
    tag = "FBBT-box" if use_fbbt else "raw-box "
    print(
        f"  {name:<10} {tag}: status={p.status if p else None!s:<10} "
        f"lower_bound={p.lower_bound if p else None!s:<22} -> useful={useful}"
    )


def main() -> None:
    for name in INSTS:
        print(f"\n{name}:")
        probe(name, use_fbbt=False)
        probe(name, use_fbbt=True)


if __name__ == "__main__":
    main()
