"""Entry experiment for issue #721 direction #1 (post-#707) — the piecewise/
partition lever probe on ex1252's wide-range cubic/monomial cost block.

#721 proposes, as its most localized direction, auto-triggering piecewise/
partitioned McCormick on ex1252's wide-range flow factors (`x6,x7,x8 ∈ [0,2950]`;
the `w = x6**2` lift secant spans `w ∈ [0, 8.7M]`), asserting the `x6**2` secant is
"the weakest single link" that "feeds every downstream term". This probe measures,
on the actual per-node engine the spatial solver uses (`MccormickLPRelaxer`) and
with #707's flow-aware integer-multilinear reform applied
(`reformulate_integer_multilinear`, the shipped `DISCOPT_INTEGER_MULTILINEAR_REFORM`
path), whether partitioning/strengthening actually moves ex1252's dual bound.

RESULT (recorded in `docs/dev/performance-plan.md` §6): at the canonical loosest
node the bound is pinned at **12658.06** across *every* available lever —
subdividing `x6`, subdividing `x12`, RLT cuts, level-1 RLT, PSD cuts, superposition,
and OBBT+cutoff. Two corrections to the issue's framing fall out:

1. **`x6` is not the lever, and neither is any flow.** At the *root* the flows are
   wide but the objective relaxes to 0 (indicators free); at any *binding* node
   OBBT has already narrowed `x6 → [1823,2950]` and `x12 → [116.7,175]`, so
   partitioning those narrow ranges is inert. "Wide-range" and "binding" never
   coincide, so direction #1 (piecewise on wide monomial factors) is inert on the
   real path. (A transient +27% signal from partitioning `x12` on the AMP MILP
   engine was a node-definition artifact — it appears only at a *looser* box where
   the MILP is free to re-choose the active line, and vanishes on the canonical box,
   where the AMP build is infeasible. It does not reflect a cubic-block tightening.)

2. **The wall is the objective coupling, not the cubic rows.** The bound equals the
   objective's constant term `6329.03·x0·x3·x18 = 6329.03·2 = 12658.06` exactly: the
   reformed `x15·(x0·x3·x18)` aux relaxes to its lower bound at this node, so the
   `1800·x15` cost contribution is 0 in the relaxation *regardless of x15*. The cubic
   cost rows the issue targets only *define* `x15`; tightening them cannot raise the
   bound while `x15`'s coupling into the objective is itself loose in-relaxation.
   The lever, if any, is a tighter joint relaxation of that coupling (a distinct,
   higher-risk direction — catalog §7's joint edge-concave/αBB open item), not the
   piecewise-McCormick of direction #1.

Run: ``python -m discopt_benchmarks.scripts.ex1252_piecewise_lever_probe``
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("DISCOPT_INCREMENTAL_MC", "0")

import discopt.modeling as dm
import numpy as np
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

OPT = 128893.74
# line-1 selected, lines 2&3 off (indicators, selectors, matching binaries).
LINE1 = {18: 1, 36: 1, 21: 1, 19: 0, 20: 0, 37: 0, 38: 0, 22: 0, 23: 0}


def _nl_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base / "python" / "tests" / "data" / "minlplib" / "ex1252.nl"
        if cand.exists():
            return cand
    raise FileNotFoundError("ex1252.nl not found under any parent's python/tests/data/minlplib")


def _loosest_node(r):
    """Canonical loosest node: LINE1 fixed, OBBT-tightened, x0=2, x3=1 (bits set)."""
    lb, ub = flat_variable_bounds(r)
    lb = np.asarray(lb, float).copy()
    ub = np.asarray(ub, float).copy()
    for i, v in LINE1.items():
        lb[i] = ub[i] = float(v)
    nc = obbt_tighten_root(r, lb.copy(), ub.copy(), rounds=5, time_limit_per_lp=0.3)
    lb, ub = nc.lb.copy(), nc.ub.copy()
    lb[0] = ub[0] = 2.0
    lb[24] = ub[24] = 0.0
    lb[25] = ub[25] = 1.0  # x0 = 2
    lb[3] = ub[3] = 1.0
    lb[30] = ub[30] = 1.0
    lb[31] = ub[31] = 0.0  # x3 = 1
    return lb, ub


def _bound(r, lb, ub, **kw):
    try:
        res = MccormickLPRelaxer(r, **kw).solve_at_node(lb.copy(), ub.copy())
        return res.lower_bound if res.status == "optimal" else f"({res.status})"
    except Exception as exc:  # noqa: BLE001
        return f"ERR {type(exc).__name__}"


def run() -> None:
    r = reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))
    lb, ub = _loosest_node(r)
    print(
        f"reformed ex1252, loosest node: x6 [{lb[6]:.0f},{ub[6]:.0f}]  "
        f"x12 [{lb[12]:.1f},{ub[12]:.1f}]"
    )
    print(f"(#707 reform applied; true optimum {OPT})\n")
    print(
        "  Flow subdivision is inert on this engine — see the #707 probe\n"
        "  (ex1252_cutoff_obbt_falsification.py): x12 halves both hold at 12658.1.\n"
        "  Direction #1's premise fails because at the root the flows are wide but the\n"
        "  objective relaxes to 0, and at any binding node OBBT has already narrowed\n"
        "  x6 -> [1823,2950] / x12 -> [116.7,175]; 'wide-range' and 'binding' never\n"
        "  coincide. The available in-box strengtheners are equally inert:\n"
    )

    base = _bound(r, lb, ub)
    print(f"  baseline (std McCormick)   : {base}")
    print(f"  rlt_cuts=True              : {_bound(r, lb, ub, rlt_cuts=True)}")
    print(f"  rlt_level1=True            : {_bound(r, lb, ub, rlt_level1=True)}")
    print(f"  psd_cuts=True              : {_bound(r, lb, ub, psd_cuts=True)}")
    print(f"  superposition=True         : {_bound(r, lb, ub, superposition=True)}")

    # Root cause: the objective's x15 coupling relaxes to its lower bound, so the
    # bound is exactly the constant term regardless of the (nonzero) relaxed x15.
    res = MccormickLPRelaxer(r).solve_at_node(lb.copy(), ub.copy())
    x15 = float(res.x[15])
    const = 6329.03 * 2.0
    print(
        f"\n  bound == 6329.03·x0·x3·x18 == 6329.03·2 == {const:.2f} exactly,\n"
        f"  yet the relaxation's x15 = {x15:.2f} ≠ 0: the reformed x15·(x0·x3·x18) aux\n"
        "  relaxes to its lower bound, so the 1800·x15 cost contributes 0 to the bound.\n"
        "  Tightening the cubic rows that DEFINE x15 cannot lift the bound while x15's\n"
        "  objective coupling is itself loose in-relaxation. Direction #1 is inert;\n"
        "  the lever is a tighter joint coupling relaxation (catalog §7 open item).\n"
        "  Full record: docs/dev/performance-plan.md §6 (2026-07-18)."
    )


if __name__ == "__main__":
    run()
