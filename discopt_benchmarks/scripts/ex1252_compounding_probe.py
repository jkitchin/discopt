"""Compounding probe for the ex1252 certification plan (issue #721 follow-up).

With the objective-coupling RLT ON (``DISCOPT_MULTILINEAR_COUPLING_RLT``), do the
previously-inert levers (flow subdivision, cutoff-OBBT) now bite at the config
node? Mechanism under test: with the RLT, the node bound = 12658.06 +
~3600*min(x15), so anything that lifts the cubic block's floor on x15
(subdivision, OBBT range reduction) now moves the BOUND — whereas before the RLT
x15 was decoupled and both levers were measured fully inert
(``ex1252_cutoff_obbt_falsification.py``, ``ex1252_piecewise_lever_probe.py``).
Certification at this node needs min x15: 12.44 -> 32.28.

Measured result (2026-07-18, recorded in docs/dev/ex1252-certification-plan.md):
  * x6 subdivision now lifts child bounds 57435 -> 62071/66932/92706 (monotone
    in x6; pre-RLT provably 0 movement) — proved 4-way min-child bound 62071.
  * OBBT now pins x12 to exactly 175.0, caps x15 at 30.89 within seconds, and at
    rounds=8 proves the whole config box EMPTY (``infeasible=True`` — the config
    is pruned outright). Pre-RLT: no movement at all.
  * The first run of this probe also exposed the Stage-1 engine fragility (child
    bounds collapsing to a 0.0 floor via a directional-widening bug in the
    conditioning clamp; crossed boxes crashing to ``error``) — both fixed, see
    the plan doc's Stage 1 record and ``python/tests/test_narrow_box_bounds.py``.

Run: ``python -m discopt_benchmarks.scripts.ex1252_compounding_probe``
"""

from __future__ import annotations

import os

os.environ["DISCOPT_MULTILINEAR_COUPLING_RLT"] = "1"

import discopt.modeling as dm
import numpy as np
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

OPT = 128893.74
LINE1 = {18: 1, 36: 1, 21: 1, 19: 0, 20: 0, 37: 0, 38: 0, 22: 0, 23: 0}

r = reformulate_integer_multilinear(dm.from_nl("python/tests/data/minlplib/ex1252.nl"))
lb, ub = flat_variable_bounds(r)
lb = np.asarray(lb, float).copy()
ub = np.asarray(ub, float).copy()
for i, v in LINE1.items():
    lb[i] = ub[i] = float(v)
nc = obbt_tighten_root(r, lb.copy(), ub.copy(), rounds=5, time_limit_per_lp=0.3)
lb, ub = nc.lb.copy(), nc.ub.copy()
lb[0] = ub[0] = 2.0
lb[24] = ub[24] = 0.0
lb[25] = ub[25] = 1.0
lb[3] = ub[3] = 1.0
lb[30] = ub[30] = 1.0
lb[31] = ub[31] = 0.0

relaxer = MccormickLPRelaxer(r)


def bd(lo, hi):
    res = relaxer.solve_at_node(lo.copy(), hi.copy())
    return res.lower_bound if res.status == "optimal" else f"({res.status})"


base = bd(lb, ub)
print(f"[A0] deep node, RLT ON, baseline bound          : {base}")
print(f"     (cert at this node needs 12658 + 3600*32.28 = {12658.06 + 3600 * 32.28:.0f})")

# A: subdivision of the flows — min over children = proved bound after branching.
for var, name, K in [(12, "x12", 2), (12, "x12", 4), (12, "x12", 8), (6, "x6", 4)]:
    edges = np.linspace(float(lb[var]), float(ub[var]), K + 1)
    kids = []
    for i in range(K):
        lo, hi = lb.copy(), ub.copy()
        lo[var], hi[var] = edges[i], edges[i + 1]
        b = bd(lo, hi)
        kids.append(b)
    finite = [b for b in kids if isinstance(b, float)]
    mn = min(finite) if finite else None
    shown = [f"{b:.0f}" if isinstance(b, float) else b for b in kids]
    print(f"[A] subdivide {name} into {K}: proved bound = {mn}   children={shown}")

# B: cutoff-driven OBBT at the deep node — inert pre-RLT; does it bite now?
# NOTE (#732 Stage 1): with more rounds OBBT's tightening can CROSS a bound —
# that is a proof the config box is EMPTY, flagged via ``res.infeasible`` (the
# solver's call sites prune on it; a crossed box handed onward now yields the
# correct ``infeasible`` verdict from solve_at_node rather than a build crash).
for cutoff, tag in [(None, "no-cutoff"), (OPT, "cutoff=opt")]:
    res = obbt_tighten_root(
        r, lb.copy(), ub.copy(), rounds=8, time_limit_per_lp=0.3, incumbent_cutoff=cutoff
    )
    b2 = "CONFIG PRUNED (box empty)" if res.infeasible else bd(res.lb, res.ub)
    print(
        f"[B] OBBT({tag:10s}): x6[{res.lb[6]:.0f},{res.ub[6]:.0f}] "
        f"x12[{res.lb[12]:.1f},{res.ub[12]:.1f}] x15[{res.lb[15]:.2f},{res.ub[15]:.2f}] "
        f"infeasible={res.infeasible} -> {b2}"
    )

# B2: compounding loop — OBBT(cutoff) then subdivide x12 on the tightened box.
res = obbt_tighten_root(
    r, lb.copy(), ub.copy(), rounds=8, time_limit_per_lp=0.3, incumbent_cutoff=OPT
)
if res.infeasible:
    print("[B2] OBBT(cutoff) proves the config box empty — node pruned outright.")
else:
    l2, u2 = res.lb.copy(), res.ub.copy()
    edges = np.linspace(float(l2[12]), float(u2[12]), 5)
    kids = []
    for i in range(4):
        lo, hi = l2.copy(), u2.copy()
        lo[12], hi[12] = edges[i], edges[i + 1]
        kids.append(bd(lo, hi))
    finite = [b for b in kids if isinstance(b, float)]
    mn = min(finite) if finite else None
    print(f"[B2] OBBT(cutoff) + subdivide x12 into 4: proved bound = {mn}")
