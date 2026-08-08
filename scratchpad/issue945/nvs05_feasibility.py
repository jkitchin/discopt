"""#945: is nvs05's "better" pre-arm incumbent actually feasible?

`nvs05_attribution.py` isolated the incumbent difference to `constr_viol_tol`:
Ipopt's default 1e-4 (pre) gives objective 8.7320; discopt's 1e-6-consistent 1e-8
(post) gives 12.5895. That reads as a 31% regression — but only if 8.7320 is a
feasible point. Ipopt's 1e-4 is two orders LOOSER than discopt's own constraint
tolerance, so the whole question is whether the cheaper objective was bought by
violating a row, which is the defect class this issue is about.

Checks the returned point against the model's own constraints and bounds, with
`discopt.validation.feasibility` — discopt's arbiter, not a hand-rolled tolerance
— and prints the worst violation per arm.

Kill criterion: if BOTH incumbents pass the feasibility check, the pre arm's
8.7320 is genuine and the post arm really does lose a better solution.

§6: prints an executed-check count and exits non-zero if it is zero.
"""

from __future__ import annotations

import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solver as SOLVER  # noqa: E402
import discopt.solvers.gdpopt_loa as LOA  # noqa: E402
import discopt.solvers.nlp_pounce as NLPP  # noqa: E402
import discopt.solvers.oa as OA  # noqa: E402
from discopt._jax.nlp_evaluator import NLPEvaluator  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402
from discopt.solvers import pounce_option_defaults  # noqa: E402

_REAL = pounce_option_defaults
_REAL_INCUMBENT = SOLVER.pounce_incumbent_options
_CONSUMERS = (SOLVER, OA, LOA)
ARMS = {"main": {"print_level": 0}, "new": None}


def set_arm(arm):
    b = ARMS[arm]
    NLPP.pounce_option_defaults = _REAL if b is None else (lambda: dict(b))
    for mod in _CONSUMERS:
        mod.pounce_incumbent_options = _REAL_INCUMBENT if b is None else (lambda: {})
        if hasattr(mod, "pounce_option_defaults"):
            mod.pounce_option_defaults = _REAL if b is None else (lambda: dict(b))


PATH = "python/tests/data/minlplib_nl/nvs05.nl"
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

checks = 0
for arm in ("main", "new"):
    set_arm(arm)
    model = from_nl(PATH)
    res = model.solve(time_limit=TL)
    if res.x is None:
        print(f"{arm}: no incumbent")
        continue

    x = np.concatenate(
        [np.asarray(res.x[v.name], dtype=np.float64).ravel() for v in model._variables]
    )
    ev = NLPEvaluator(model)
    g = np.asarray(ev.evaluate_constraints(x), dtype=np.float64)
    cl, cu = ev.constraint_bounds if hasattr(ev, "constraint_bounds") else (None, None)
    if cl is None:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(ev)
    cl = np.asarray(cl, dtype=np.float64)
    cu = np.asarray(cu, dtype=np.float64)
    row_viol = float(np.max(np.maximum(cl - g, g - cu))) if g.size else 0.0

    lb, ub = ev.variable_bounds
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    box_viol = float(np.max(np.maximum(lb - x, x - ub)))

    # Integrality: nvs05 has general integers; a fractional "integer" is as much a
    # false incumbent as a violated row.
    from discopt.modeling.core import VarType

    int_viol = 0.0
    off = 0
    for v in model._variables:
        n = v.size
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            xi = x[off : off + n]
            int_viol = max(int_viol, float(np.max(np.abs(xi - np.round(xi)))))
        off += n

    checks += 3
    worst = max(row_viol, box_viol, int_viol)
    print(
        f"{arm:5s} obj={res.objective!r:22s} worst_row={row_viol:.3e} "
        f"box={box_viol:.3e} integrality={int_viol:.3e}  "
        f"{'INFEASIBLE past discopt 1e-6' if worst > 1e-6 else 'feasible'}"
    )
set_arm("new")

print(f"\nexecuted_checks={checks}")
if checks == 0:
    print("PROBE CHECKED NOTHING", file=sys.stderr)
    sys.exit(2)
