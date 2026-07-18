"""#732 Stage 2 deliverable-2 entry experiment: pump-count zero-dichotomy recursion.

Multi-line configs bound at ~0 because free pump counts let each line's cost
coupling sit at its x_i=0 corner. Split every active-line pump var into {0} vs
[1,3] (a partition of the implied-integer domain): with lb >= 1 the bit-link RLT
McCormick chains get real lower corners, and the x=0 children degenerate toward
the (already strongly bounded) fewer-line configs. Measure min over the 16
children of config (1,1,0) — the proved recursed bound for that config.

MEASURED (2026-07-18, recorded in docs/dev/ex1252-certification-plan.md Stage 2):
  * 12/16 children PRUNE (LP-infeasible) — the dichotomy eliminates hard;
  * 2 children are `numerical` (x0=x3=0 with line 1 on: genuinely infeasible by
    the head equation x18 = 0.0025*x9*x3, but the ill-conditioned LP cannot
    Farkas-certify it and OBBT rounds=8 does not cross — the SAME contradiction
    is provable by interval FBBT alone, so the disjunctive pass must run a
    per-box FBBT step (Rust binding: `tighten_var_bounds` + `fbbt`/
    `fbbt_with_cutoff`) before the LP;
  * 2 children bound: (1+,1+,0,1+) = 62027 and (1+,1+,1+,1+) = 6814 — the
    all-pumps-on child needs one more value-split level (x in {1},{2},{3}) to
    climb, since lb=1 corners keep the multilinear chain near its constant.
"""

import os

os.environ["DISCOPT_MULTILINEAR_COUPLING_RLT"] = "1"

import itertools

import discopt.modeling as dm
import numpy as np
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

INCUMBENT = 134471.56

r = reformulate_integer_multilinear(dm.from_nl("python/tests/data/minlplib/ex1252.nl"))
lb0, ub0 = flat_variable_bounds(r)
lb0 = np.asarray(lb0, float).copy()
ub0 = np.asarray(ub0, float).copy()

# Config (1,1,0): lines 1,2 on; line 3 off. Pump vars: line1 (x0 par, x3 ser),
# line2 (x1 par, x4 ser). Line-3 vars pinned 0 by gating.
for ind, sel, v in ((18, 36, 1.0), (19, 37, 1.0), (20, 38, 0.0)):
    lb0[ind] = ub0[ind] = v
    lb0[sel] = ub0[sel] = v

PUMPS = [0, 3, 1, 4]  # x0, x3 (line 1), x1, x4 (line 2)


def child_bound(dichotomy):
    lb, ub = lb0.copy(), ub0.copy()
    for var, hi_side in zip(PUMPS, dichotomy, strict=True):
        if hi_side:
            lb[var] = max(lb[var], 1.0)
        else:
            ub[var] = min(ub[var], 0.0)
    res = obbt_tighten_root(
        r, lb.copy(), ub.copy(), rounds=3, time_limit_per_lp=0.2, incumbent_cutoff=INCUMBENT
    )
    if res.infeasible:
        return "OBBT-pruned", None
    node = MccormickLPRelaxer(r).solve_at_node(res.lb.copy(), res.ub.copy())
    if node.status == "infeasible":
        return "LP-infeasible", None
    if node.status != "optimal":
        return node.status, None
    return "optimal", float(node.lower_bound)


alive, statuses = [], {}
for dich in itertools.product((0, 1), repeat=4):
    st, b = child_bound(dich)
    statuses[dich] = (st, b)
    tag = f"{b:.0f}" if b is not None else st
    print(f"  (x0,x3,x1,x4) {'/'.join('0' if d == 0 else '1+' for d in dich):11s}: {tag}")
    if b is not None:
        alive.append(b)

print(
    f"\nconfig (1,1,0) recursed bound = min over children = {min(alive):.1f}"
    if alive
    else "all children pruned!"
)
print("(pre-recursion config bound: ~0; single-line configs: 71644-115466)")
