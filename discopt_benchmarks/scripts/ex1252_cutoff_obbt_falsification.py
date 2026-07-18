"""Reproduction: issue #707 falsification — "cutoff-driven range reduction
certifies the ex1252 class" (see docs/dev/performance-plan.md §6).

The #707 flow-aware integer-multilinear envelope lifts ex1252's dual bound off its
5134 floor but does not certify it (~48k vs opt 128893.74). This probe tests
whether the *incumbent objective cutoff* is the missing lever: it hands OBBT the
known optimum as a cutoff and measures whether it shrinks the wide continuous
flows x6/x12 and lifts the dual — at the root, on a line-selected box, under
continuous subdivision, and with the integer flow factors fixed.

Result: the cutoff is inert at every level (the relaxation is ~10x looser than the
cutoff, so it never binds). The bottleneck is the continuous-cubic relaxation
strength, not range reduction. Run:

    python -m discopt_benchmarks.scripts.ex1252_cutoff_obbt_falsification
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("DISCOPT_INCREMENTAL_MC", "0")  # cold build per call (no patch-state artifact)

import discopt.modeling as dm
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

OPT = 128893.74
# line-1 selected, lines 2&3 off (indicators, selectors, matching binaries)
LINE1 = {18: 1, 36: 1, 21: 1, 19: 0, 20: 0, 37: 0, 38: 0, 22: 0, 23: 0}


def _nl_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base / "python" / "tests" / "data" / "minlplib" / "ex1252.nl"
        if cand.exists():
            return cand
    raise FileNotFoundError("ex1252.nl not found under any parent's python/tests/data/minlplib")


def _bound(r, lb, ub):
    res = MccormickLPRelaxer(r).solve_at_node(lb, ub)
    return res.lower_bound if res.status == "optimal" else None


def main() -> None:
    r = reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))
    lb, ub = flat_variable_bounds(r)

    # (1) root box
    print(f"root, OBBT no-cutoff : {_root_obbt_bound(r, lb, ub, None):.1f}")
    print(f"root, OBBT cutoff=opt: {_root_obbt_bound(r, lb, ub, OPT):.1f}  (cutoff inert)")

    # (2) line-1 selected box
    lb1, ub1 = lb.copy(), ub.copy()
    for i, v in LINE1.items():
        lb1[i] = ub1[i] = float(v)
    nc = obbt_tighten_root(r, lb1.copy(), ub1.copy(), rounds=5, time_limit_per_lp=0.3)
    cc = obbt_tighten_root(r, lb1.copy(), ub1.copy(), rounds=5, time_limit_per_lp=0.3,
                           incumbent_cutoff=OPT)
    print(f"\nline-1 box: x6 [{nc.lb[6]:.0f},{nc.ub[6]:.0f}] x12 [{nc.lb[12]:.1f},{nc.ub[12]:.1f}]")
    print(f"line-1, OBBT no-cutoff : {_bound(r, nc.lb, nc.ub):.1f}")
    print(f"line-1, OBBT cutoff=opt: {_bound(r, cc.lb, cc.ub):.1f}  (identical -> cutoff inert)")

    # (3) subdivide continuous x12 (fresh cold relaxers) — bound does not move
    lb2, ub2 = nc.lb.copy(), nc.ub.copy()
    mid = 0.5 * (lb2[12] + ub2[12])
    lo, hi = lb2.copy(), ub2.copy(); hi[12] = mid
    print(f"\nx12 lower half: {_bound(r, lo, hi):.1f}")
    lo, hi = lb2.copy(), ub2.copy(); lo[12] = mid
    print(f"x12 upper half: {_bound(r, lo, hi):.1f}  (subdividing flows does not tighten)")

    # (4) fix the integer flow factors: the loosest node is the binding one
    print("\nfixing integer (x0,x3) on the line-1 box:")
    best = float("inf")
    for x0 in (1, 2, 3):
        for x3 in (1, 2, 3):
            lo, hi = nc.lb.copy(), nc.ub.copy()
            lo[0] = hi[0] = x0; lo[3] = hi[3] = x3
            lo[24] = hi[24] = x0 & 1; lo[25] = hi[25] = (x0 >> 1) & 1
            lo[30] = hi[30] = x3 & 1; lo[31] = hi[31] = (x3 >> 1) & 1
            b = _bound(r, lo, hi)
            tag = f"{b:.0f}" if b is not None else "infeas"
            print(f"    x0={x0} x3={x3}: {tag}")
            if b is not None:
                best = min(best, b)
    print(f"min over integer (x0,x3) = {best:.0f}  vs opt {OPT}  (~10x loose -> relaxation is the wall)")


def _root_obbt_bound(r, lb, ub, cutoff):
    res = obbt_tighten_root(r, lb.copy(), ub.copy(), rounds=5, time_limit_per_lp=0.3,
                            incumbent_cutoff=cutoff)
    return _bound(r, res.lb, res.ub)


if __name__ == "__main__":
    main()
