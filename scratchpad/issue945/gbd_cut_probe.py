"""#945 / GBD: attribute the stalled bound to the recourse point at y=0.

Hypothesis: at the master proposal y=0 the recourse constraint `x0^2+x1^2 <= 0`
forces x=0, where its Jacobian `2x` is also 0 — the LICQ fails and the
Lagrangian's recourse gradient never cancels. Pre-#945, Ipopt's
`bound_relax_factor` let the point settle at x ~ 1e-4 instead, where the Jacobian
is nonzero and the cut came out usable. If so, the cut at y=0 collapses to the
box-minimum anchor and GBD stops making progress.

Kill criterion: if the y=0 recourse point is x=0 in BOTH arms, or if the cut
anchor is the same in both, the hypothesis is wrong.

Records every recourse solve in both arms; prints an executed-call count.
"""

from __future__ import annotations

import inspect
import sys

import numpy as np

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as NLPP
from discopt.decomposition.benders import solve_benders
from discopt.solvers import pounce_option_defaults

assert "opts = pounce_option_defaults()" in inspect.getsource(NLPP.solve_nlp), "not the #945 tree"

_PRE = {"print_level": 0}
_REAL = pounce_option_defaults
_ORIG = NLPP.solve_nlp

calls: list[tuple] = []
_ARM = "pre"


def _tap(evaluator, x0, *a, **k):
    res = _ORIG(evaluator, x0, *a, **k)
    if res.x is not None:
        x = np.asarray(res.x, dtype=np.float64)
        try:
            g = np.asarray(evaluator.evaluate_constraints(x), dtype=np.float64)
            J = np.asarray(evaluator.evaluate_jacobian(x), dtype=np.float64)
            jnorm = float(np.max(np.abs(J))) if J.size else 0.0
        except Exception as exc:  # recorded, never swallowed (§7)
            g, jnorm = f"ERR {exc}", float("nan")
        mu = None if res.multipliers is None else np.asarray(res.multipliers, dtype=np.float64)
        calls.append((_ARM, str(res.status), x.copy(), g, jnorm, mu, res.objective))
    return res


NLPP.solve_nlp = _tap


def build():
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


for arm in ("pre", "post"):
    _ARM = arm
    NLPP.pounce_option_defaults = (lambda: dict(_PRE)) if arm == "pre" else _REAL
    r = solve_benders(build(), time_limit=30)
    print(f"\n=== arm={arm}: status={r.status} obj={r.objective!r} bound={r.bound!r}")
NLPP.pounce_option_defaults = _REAL

print(f"\n{'arm':5s} {'status':10s} {'x':>34s} {'g(x)':>16s} {'max|J|':>10s} {'mu':>14s}")
print("-" * 96)
for arm, status, x, g, jnorm, mu, obj in calls:
    xs = np.array2string(x, precision=6, max_line_width=200)
    gs = np.array2string(np.atleast_1d(g), precision=6) if not isinstance(g, str) else g
    ms = "None" if mu is None else np.array2string(mu, precision=4)
    print(f"{arm:5s} {status:10s} {xs:>34s} {gs:>16s} {jnorm:10.3e} {ms:>14s}")

print(f"\nexecuted_recourse_solves={len(calls)}")
if not calls:
    print("PROBE TAPPED NOTHING", file=sys.stderr)
    sys.exit(2)
