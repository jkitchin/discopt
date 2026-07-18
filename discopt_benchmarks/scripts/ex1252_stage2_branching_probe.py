"""#732 Stage 2 entry experiments: configuration-first branching on ex1252.

Three experiments, run 2026-07-18 (reform + coupling RLT ON throughout); results
recorded in ``docs/dev/ex1252-certification-plan.md`` Stage 2.

A. **Instrument the 400-node tree** (branch-count sink): the tree spends its
   branching on the reform's *continuous* big-M product auxes (``_ipx_v*`` — top
   column 29 branchings) whose subdivision provably cannot pin the coupling,
   while the priority set (``DISCOPT_OBJ_BRANCH_PRIORITY``) contains only 6 vars
   post-reform and the top priority var got 4 branchings. The ``_ipx_e``
   expansion bits — whose fractionality is the L2 leak — appear only in linear
   big-M rows, invisible to the nonlinear-term-keyed priority detector.

B. **All-binaries priority (monkeypatch)**: forcing every binary into the
   priority set FAILED the kill criterion — 400-node global dual 12658 (< the
   16304 baseline, << the 33k gate), incumbent worse. Root cause: the priority
   hint is *fractionality-gated*, and the node LP vertex often has integral
   binaries while the coupling stays loose through the v-auxes — no fractional
   signal, no hint, and the standard selector branches the useless v columns.
   Hint-based configuration-first branching is falsified for this tree.

C. **Disjunctive route (the plan's recorded alternative)**: enumerate the 2^3
   line-indicator patterns (every feasible point has integral indicators — they
   are equality-pinned to binaries), OBBT each config box, take the min of
   per-config bounds as a valid root bound. Measured:

     lines (0,0,1): 90592  (PRUNED outright under the incumbent cutoff)
     lines (0,1,1): 59806
     lines (1,0,0): 38524
     lines (1,0,1)/(1,1,0)/(1,1,1): ~0   <-- the residual wall
     (0,0,0): infeasible; (0,1,0): numerical

   Single-line configs certify far above the tree's 16.3k global dual; the
   multi-line configs re-create the L2 decoupling one level down: their coupling
   runs through the pump-count integer-BILINEAR products (``x9*x3 = 400*x18``
   etc.), which the reform expands on its *bilinear* path — initially not
   covered by the #721 coupling RLT. Extending the bit-link RLT to that path
   (Stage 2 deliverable 1, same flag) amplifies the per-config bounds to
   (0,0,1) 115466 / (0,1,1) 90429 / (1,0,0) 71644 — the numbers this script now
   reproduces — leaving only the multi-line configs weak (pump-count recursion,
   deliverable 2).

Run: ``python -m discopt_benchmarks.scripts.ex1252_stage2_branching_probe``
(runs experiment C, the cheap decisive one; A/B need a full 400-node solve).
"""

from __future__ import annotations

import itertools
import os

os.environ["DISCOPT_MULTILINEAR_COUPLING_RLT"] = "1"

import discopt.modeling as dm
import numpy as np
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

OPT = 128893.74
INCUMBENT = 134471.56  # found by discopt itself (RLT ON) — a legitimate cutoff


def _nl_path():
    from pathlib import Path

    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base / "python" / "tests" / "data" / "minlplib" / "ex1252.nl"
        if cand.exists():
            return cand
    raise FileNotFoundError("ex1252.nl not found")


def run() -> None:
    r = reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))
    lb0, ub0 = flat_variable_bounds(r)
    lb0 = np.asarray(lb0, float).copy()
    ub0 = np.asarray(ub0, float).copy()

    def config_bound(pattern, cutoff):
        lb, ub = lb0.copy(), ub0.copy()
        for (ind, sel), v in zip(((18, 36), (19, 37), (20, 38)), pattern, strict=True):
            lb[ind] = ub[ind] = float(v)
            lb[sel] = ub[sel] = float(v)
        res = obbt_tighten_root(
            r, lb.copy(), ub.copy(), rounds=6, time_limit_per_lp=0.3, incumbent_cutoff=cutoff
        )
        if res.infeasible:
            return "OBBT-pruned", None
        node = MccormickLPRelaxer(r).solve_at_node(res.lb.copy(), res.ub.copy())
        if node.status == "infeasible":
            return "LP-infeasible", None
        if node.status != "optimal":
            return node.status, None
        return "optimal", float(node.lower_bound)

    for cutoff, tag in [(None, "no cutoff"), (INCUMBENT, f"cutoff={INCUMBENT:.0f}")]:
        print(f"\n=== per-config bounds ({tag}) ===")
        alive = []
        for pattern in itertools.product((0, 1), repeat=3):
            st, b = config_bound(pattern, cutoff)
            print(f"  lines {pattern}: {st:14s} bound={b}")
            if b is not None:
                alive.append(b)
        if alive:
            print(
                f"  DISJUNCTIVE ROOT BOUND = min over configs = {min(alive):.1f} "
                f"(tree global dual at 400 nodes: 16304; opt {OPT})"
            )


if __name__ == "__main__":
    run()
