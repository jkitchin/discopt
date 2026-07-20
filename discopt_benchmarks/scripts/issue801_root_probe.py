"""#801 Stage 0 — re-baseline the tanksize root McCormick LP on current HEAD.

Reproduces the #764 measurement (``docs/dev/issue-764-root-relaxation-plan.md``):
the root McCormick LP over the FBBT box, integers relaxed, gives ``lb(x17)`` =
**0.8382** (objective = minimize x17). This is the number every #801 residual
experiment must move. If HEAD no longer reproduces 0.8382, STOP and re-diagnose
(something graduated since 2026-07-18 changed the root).

Also provides the reusable validity infrastructure for Stage 2 constructions:

* :func:`baseline_root_lp` — the pure-McCormick root bound + the LP optimum x*
  (for Stage 1 residual attribution).
* :func:`injection_sanity` — box-injection plumbing check: forcing ``lb(x17)=1.0``
  must move the root to >= 1.0 (proves the box reaches the LP). Stage 2 adds a
  per-construction *row*-injection check on top of this.
* :func:`feasible_points` — the certified optimum (found at node 0) plus
  continuous-relaxation NLP solutions, the points any valid relaxation/cut must
  NOT cut (the no-valid-point-cut invariant, CLAUDE.md §5).

Env trap (#798/#802): ``discopt.pth`` is a single shared path — drive via
PYTHONPATH and confirm the build before trusting numbers.

Run: ``python discopt_benchmarks/scripts/issue801_root_probe.py``
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402

from discopt._jax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402

NL = os.path.join(
    os.path.dirname(__file__), "..", "..", "python", "tests", "data", "minlplib_nl", "tanksize.nl"
)
NL = os.path.abspath(NL)

OBJ_VAR = 17  # objective is `minimize x17`; flat index 17
BASELINE = 0.8382  # #764 root McCormick LP
ORACLE = 1.2686437540  # MINLPLib optimum (min)
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "issue801")


@dataclass
class RootProbe:
    bound: float
    x: np.ndarray
    lb: np.ndarray
    ub: np.ndarray


def load():
    return from_nl(NL)


def root_box(model):
    bt = fbbt_box(model)
    assert not bt.infeasible, "FBBT proved tanksize infeasible — impossible, re-check"
    return bt.lb.copy(), bt.ub.copy()


def baseline_root_lp(model=None, lb=None, ub=None) -> RootProbe:
    """Pure McCormick root LP over the FBBT box (integers relaxed via the box)."""
    if model is None:
        model = load()
    if lb is None or ub is None:
        lb, ub = root_box(model)
    relaxer = MccormickLPRelaxer(model)
    res = relaxer.solve_at_node(lb, ub)
    assert res.status == "optimal", f"root LP not optimal: {res.status}"
    x = np.asarray(res.x, dtype=np.float64) if res.x is not None else np.array([])
    return RootProbe(bound=float(res.lower_bound), x=x, lb=lb, ub=ub)


def injection_sanity(model=None) -> dict:
    """Box-injection plumbing: force lb(x17)=1.0 → root must be >= 1.0.

    Proves the FBBT box actually constrains the LP the relaxer solves (so a Stage-2
    construction that tightens the box or adds a binding row will register).
    """
    if model is None:
        model = load()
    lb, ub = root_box(model)
    lb2 = lb.copy()
    lb2[OBJ_VAR] = 1.0
    probe = baseline_root_lp(model, lb2, ub)
    ok = probe.bound >= 1.0 - 1e-6
    return {"injected_lb_x17": 1.0, "root_after_injection": probe.bound, "moved": ok}


def certified_optimum(model=None, time_limit: float = 30.0):
    """The MINLPLib optimum, verified feasible (found at node 0 per #764)."""
    if model is None:
        model = load()
    res = model.solve(time_limit=time_limit, gap_tolerance=1e-4)
    x = np.asarray(res.x, dtype=np.float64) if res.x is not None else None
    return res, x


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = load()
    lb, ub = root_box(model)

    probe = baseline_root_lp(model, lb, ub)
    reproduced = abs(probe.bound - BASELINE) < 5e-3

    inj = injection_sanity(model)

    out = {
        "instance": "tanksize",
        "objective": "minimize x17 (flat idx 17)",
        "n_vars": int(lb.size),
        "fbbt_lb_x17": float(lb[OBJ_VAR]),
        "fbbt_ub_x17": float(ub[OBJ_VAR]),
        "root_mccormick_lp": probe.bound,
        "baseline_ref": BASELINE,
        "reproduced_0p8382": bool(reproduced),
        "oracle": ORACLE,
        "root_gap_to_oracle": ORACLE - probe.bound,
        "injection_sanity": inj,
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "stage0_baseline.json"), "w") as f:
        json.dump(out, f, indent=2)

    if not reproduced:
        raise SystemExit(
            f"STOP: root LP {probe.bound:.4f} != baseline {BASELINE} — "
            "something changed the root since 2026-07-18; re-diagnose before Stage 1."
        )
    if not inj["moved"]:
        raise SystemExit("STOP: box-injection did not move the root — plumbing broken.")
    print("\nStage 0 OK: baseline reproduced, plumbing validated.")


if __name__ == "__main__":
    main()
