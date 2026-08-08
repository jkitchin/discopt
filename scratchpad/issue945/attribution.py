"""#945 attribution probe: what actually changes when the NLP path stops
returning points outside their declared box.

For each affected fixture, runs BOTH arms in-process and reports
status / objective / bound / gap, plus two *independent* soundness checks of the
returned incumbent that do not go through the solver's own guards:

  * ``box_viol``  — max distance of x outside its declared [lb, ub]
  * ``cert_viol`` — max(0, bound - incumbent): the certificate invariant
                    ``bound <= incumbent`` for a minimization (CLAUDE.md §1)

Prints an executed-comparison count and exits non-zero if it measured nothing
(CLAUDE.md §6). Arms are interleaved per fixture, not run as two sequential
blocks, so a drifting machine cannot masquerade as an arm effect (§9).

Usage:  python -u scratchpad/issue945/attribution.py
"""

from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
from discopt.solvers import pounce_option_defaults

# §8: assert which code is under test before measuring anything.
_SEEDED = "opts = pounce_option_defaults()" in inspect.getsource(nlp_pounce.solve_nlp)
if not _SEEDED:
    print(
        "FATAL: nlp_pounce.solve_nlp is NOT seeded from pounce_option_defaults; "
        "this probe compares arms of the #945 change and needs the seeded tree.",
        file=sys.stderr,
    )
    sys.exit(2)

# Arms. "main" is EXACTLY what `nlp_pounce.solve_nlp` did before #945: it set
# print_level only, leaving BOTH constr_viol_tol (Ipopt default 1e-4) and
# bound_relax_factor (1e-8) at Ipopt's values. Naming only one of them leaves the
# other at the shipped value and mislabels the arm (the #940 lesson) — "cvt_only"
# is kept precisely to show how much of the effect each option carries.
_ARMS = {
    "main": {"print_level": 0},
    "cvt": {"print_level": 0, "constr_viol_tol": 1e-8, "bound_relax_factor": 1e-8},
    "new": None,  # the shipped pounce_option_defaults()
}
_REAL = pounce_option_defaults


def _install(arm: str) -> None:
    override = _ARMS[arm]
    nlp_pounce.pounce_option_defaults = _REAL if override is None else (lambda: dict(override))


# ── fixtures ────────────────────────────────────────────────────────────────
# Copied from the failing tests so the probe is standalone (and so a test-file
# edit cannot silently change what this measured).


_TESTS = pathlib.Path(__file__).resolve().parents[2] / "python" / "tests"
sys.path.insert(0, str(_TESTS))
# Import the REAL fixtures rather than re-typing them: a hand-copied model is a
# different model, and the first draft of this probe measured one (its
# "mindtpy_simple" had optimum 3.0, the test's has 3.5).
from test_mip_nlp import (  # noqa: E402
    _mindtpy_constraint_qualification_example as _mindtpy_cq,
)
from test_mip_nlp import _mindtpy_simple_minlp  # noqa: E402


def _gdp_simple_disjunction(name="g"):
    m = dm.Model(name)
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.either_or([[x <= 3.0], [x >= 7.0]], name="choice")
    m.minimize(x)
    return m


# (label, builder, solve kwargs, hand-verified true optimum)
#   mindtpy_simple : optimum 3.5 (y = (0,1,0), x = (2,2))
#   mindtpy_cq     : y=0 is infeasible (x·ln x + 5 <= 0 has no root on [1,10]),
#                    so y=1 forces (x-3)^2 <= 0, i.e. x = 3 exactly. Optimum 3.0.
#   gdp            : min x over (x<=3) or (x>=7), x in [0,10]. Optimum 0.0.
_OA = dict(solver="mip-nlp", mip_nlp_method="oa", time_limit=60, max_nodes=100)
CASES = [
    ("mindtpy_simple/oa", lambda: _mindtpy_simple_minlp("a"), _OA, 3.5),
    ("mindtpy_simple/roa-L1", lambda: _mindtpy_simple_minlp("b"), {**_OA, "add_regularization": "level_L1"}, 3.5),
    ("mindtpy_cq/roa-L1", lambda: _mindtpy_cq("c"), {**_OA, "add_regularization": "level_L1"}, 3.0),
    ("mindtpy_cq/roa-Linf", lambda: _mindtpy_cq("d"), {**_OA, "add_regularization": "level_L_infinity"}, 3.0),
    ("mindtpy_cq/roa-gradlag", lambda: _mindtpy_cq("e"), {**_OA, "add_regularization": "grad_lag"}, 3.0),
    ("gdp_simple_disjunction", _gdp_simple_disjunction, dict(gdp_method="loa", time_limit=30), 0.0),
]


def _box_violation(model, res) -> float:
    """Max distance of the returned point outside its declared box."""
    worst = 0.0
    for v in model._variables:
        val = np.asarray(res.x[v.name], dtype=np.float64).ravel()
        lb = np.asarray(v.lb, dtype=np.float64).ravel()
        ub = np.asarray(v.ub, dtype=np.float64).ravel()
        lb = np.broadcast_to(lb, val.shape)
        ub = np.broadcast_to(ub, val.shape)
        worst = max(worst, float(np.max(lb - val)), float(np.max(val - ub)))
    return max(0.0, worst)


comparisons = 0
rows = []
for label, build, kwargs, true_opt in CASES:
    for arm in ("main", "cvt", "new"):  # interleaved per fixture (§9)
        _install(arm)
        model = build()
        res = model.solve(**kwargs)
        box = _box_violation(model, res)
        obj = res.objective
        bnd = res.bound
        # Certificate invariant for a minimization: bound <= incumbent.
        cert = 0.0 if (obj is None or bnd is None) else max(0.0, float(bnd) - float(obj))
        # Super-optimality: an incumbent BELOW the hand-verified optimum is,
        # by definition, not a feasible point of the declared problem.
        sup = 0.0 if obj is None else max(0.0, true_opt - float(obj))
        # A dual bound above the true optimum has cut the optimum out.
        bad_bnd = 0.0 if bnd is None else max(0.0, float(bnd) - true_opt)
        rows.append((label, arm, res.status, obj, bnd, res.gap, box, cert, sup, bad_bnd))
        comparisons += 1
_install("new")

hdr = (
    f"{'fixture':24s} {'arm':5s} {'status':9s} {'objective':>16s} {'bound':>16s} "
    f"{'gap':>11s} {'box':>9s} {'cert':>9s} {'super_opt':>9s} {'bnd>opt':>9s}"
)
print(hdr)
print("-" * len(hdr))
violations = 0
for label, arm, status, obj, bnd, gap, box, cert, sup, bad_bnd in rows:
    o = "None" if obj is None else f"{obj:.12g}"
    b = "None" if bnd is None else f"{bnd:.12g}"
    g = "None" if gap is None else f"{gap:.4e}"
    flags = []
    if box > 0:
        flags.append("OUT-OF-BOX")
    if cert > 0:
        flags.append("BOUND>INCUMBENT")
    if sup > 0:
        flags.append("SUPER-OPTIMAL")
    if bad_bnd > 0:
        flags.append("BOUND>TRUE-OPT")
    violations += len(flags)
    print(
        f"{label:24s} {arm:5s} {status:9s} {o:>16s} {b:>16s} {g:>11s} "
        f"{box:9.2e} {cert:9.2e} {sup:9.2e} {bad_bnd:9.2e}  {' '.join(flags)}"
    )

print(f"\nexecuted_comparisons={comparisons} flagged_rows={violations}")
if comparisons == 0:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(2)
