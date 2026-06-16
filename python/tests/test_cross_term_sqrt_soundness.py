"""Adversarial soundness for the cross-term sqrt lift (issue #154, increment 2).

The lifted ``sqrt(g)`` envelope (#157) force-lifts each multiplicative factor of
``g`` into a bounded McCormick product aux. Increment 2 extends the inner-factor
parser so a monomial carrying an *embedded constant* — the cross term
``2*x4*x5*x7`` in nvs05/nvs22 constraint ``C4 = sqrt(x4**2 + 2*x4*x5*x7 + x5**2)``
— lifts too: the scalar is stripped into a coefficient and the variable factors
fold into one product aux. Previously ``_lift_factor_to_col`` rejected the scalar
and the whole sqrt abstained.

Two failure modes are locked here:

1. **Unsound cut** — an envelope row that excludes a true feasible point. We pin
   the variables to interior grid points (degenerate boxes) and check the
   relaxation brackets the exact curve value from both sides. A bracket that
   holds everywhere means no cut excludes ``sqrt(g)``.

2. **Ill-conditioned LP → wrong "optimal"** — folding a cross term over a *wide*
   box yields product-aux bounds of order ``1e8``–``1e9`` (nvs05/nvs22's
   ``g`` reaches ``~1.2e9``). The fast simplex backend then returns a wrong
   "optimal" objective that *exceeds the true optimum* — an unsound dual bound
   (HiGHS still solves it correctly). The cross-term conditioning guard
   (``_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE``) must make the lift *abstain* above the
   magnitude limit, so the relaxation stays sound on every backend. The two
   MINLPLib cases are the regression: before the guard the nvs22 root bound came
   back ``8.31 > 6.0584`` (a false, super-optimal dual bound).

The invariant under test is the project's non-negotiable one: a valid lower
bound never exceeds the true optimum, and the relaxation never excludes a
feasible point.
"""

import math
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.model_utils import flat_variable_bounds
from discopt.solver import _extract_variable_info
from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

_TOL = 1e-6
_DATA = Path(__file__).parent / "data" / "minlplib"


def _cross_sqrt_model(xb, yb, zb, *, sign=1.0):
    """``sign * sqrt(x**2 + 2*x*y*z + y**2)`` over the given boxes.

    The cross term ``2*x*y*z`` is the shape increment 2 unlocks: a trilinear
    monomial with an embedded scalar ``2``.
    """
    m = dm.Model()
    x = m.continuous("x", lb=xb[0], ub=xb[1])
    y = m.continuous("y", lb=yb[0], ub=yb[1])
    z = m.continuous("z", lb=zb[0], ub=zb[1])
    m.minimize(sign * dm.sqrt(x * x + 2.0 * x * y * z + y * y))
    return m


def _root_bound(model, backend="simplex", pin=None, bounds=None):
    relaxer = MccormickLPRelaxer(model)
    relaxer._backend = backend
    if bounds is None:
        lb, ub = flat_variable_bounds(model)
    else:
        lb, ub = bounds
    lb = lb.copy()
    ub = ub.copy()
    if pin is not None:
        for i, v in pin.items():
            lb[i] = v
            ub[i] = v
    return relaxer.solve_at_node(lb, ub)


# Well-conditioned boxes: |g| stays well below the cross-term magnitude guard, so
# the lift engages and the fast simplex is reliable.
_WELL_CONDITIONED = [
    ((1.0, 3.0), (1.0, 3.0), (1.0, 3.0)),
    ((0.5, 2.0), (1.0, 4.0), (0.5, 2.5)),
    ((2.0, 5.0), (1.0, 3.0), (1.0, 2.0)),
]


@pytest.mark.parametrize("xb, yb, zb", _WELL_CONDITIONED)
def test_cross_term_sqrt_lift_engages(xb, yb, zb):
    """On a well-conditioned box the cross-term sqrt lift creates aux columns —
    i.e. it actually engages rather than silently abstaining."""
    m = _cross_sqrt_model(xb, yb, zb)
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    milp, _ = build_milp_relaxation(
        relaxer._model,
        relaxer._terms,
        relaxer._disc,
        bound_override=(np.asarray(lb), np.asarray(ub)),
        superposition=relaxer._superposition,
    )
    # 3 original variables; a lifted cross-term sqrt adds product + sqrt aux cols.
    assert len(milp._c) > 3, "cross-term sqrt lift did not engage (no aux columns)"


@pytest.mark.parametrize("xb, yb, zb", _WELL_CONDITIONED)
def test_cross_term_sqrt_encloses_curve(xb, yb, zb):
    """At every interior grid point the relaxation brackets
    ``sqrt(x**2 + 2*x*y*z + y**2)`` from both sides — the cross-term lift never
    excludes the true curve."""
    xs = [xb[0], 0.5 * (xb[0] + xb[1]), xb[1]]
    ys = [yb[0], 0.5 * (yb[0] + yb[1]), yb[1]]
    zs = [zb[0], 0.5 * (zb[0] + zb[1]), zb[1]]
    for xv in xs:
        for yv in ys:
            for zv in zs:
                true = math.sqrt(xv * xv + 2.0 * xv * yv * zv + yv * yv)
                pin = {0: xv, 1: yv, 2: zv}
                lo = _root_bound(_cross_sqrt_model(xb, yb, zb, sign=1.0), pin=pin)
                hi = _root_bound(_cross_sqrt_model(xb, yb, zb, sign=-1.0), pin=pin)
                assert lo.status == "optimal" and lo.lower_bound is not None
                assert hi.status == "optimal" and hi.lower_bound is not None
                # underestimator never rises above the curve …
                assert lo.lower_bound <= true + _TOL, (
                    f"cross-sqrt under-cut excludes curve at ({xv},{yv},{zv}): "
                    f"{lo.lower_bound} > {true}"
                )
                # … overestimator never drops below it.
                assert -hi.lower_bound >= true - _TOL, (
                    f"cross-sqrt over-cut excludes curve at ({xv},{yv},{zv}): "
                    f"{-hi.lower_bound} < {true}"
                )


@pytest.mark.parametrize("xb, yb, zb", _WELL_CONDITIONED)
def test_cross_term_sqrt_backends_agree(xb, yb, zb):
    """In the well-conditioned regime the fast simplex and HiGHS must agree — a
    disagreement would mean the lifted LP is already ill-conditioned and the fast
    bound is untrustworthy."""
    m = _cross_sqrt_model(xb, yb, zb)
    fast = _root_bound(m, backend="simplex")
    ref = _root_bound(m, backend="auto")
    assert fast.status == "optimal" and ref.status == "optimal"
    assert abs(fast.lower_bound - ref.lower_bound) <= 1e-4, (
        f"backend disagreement {fast.lower_bound} vs {ref.lower_bound} — fast bound unreliable"
    )


# nvs05/nvs22: |g| ~ 1.2e9 trips the cross-term magnitude guard. (instance, opt).
_MINLPLIB_GUARDED = [("nvs05", 5.47093), ("nvs22", 6.0584)]


@pytest.mark.parametrize("instance, optimum", _MINLPLIB_GUARDED)
@pytest.mark.parametrize("backend", ["simplex", "auto"])
def test_cross_term_guard_keeps_minlplib_sound(instance, optimum, backend):
    """The conditioning guard keeps nvs05/nvs22 root bounds sound on *both*
    backends. This is the regression: before the guard, folding C4's cross term
    over the wide FBBT box gave the fast simplex a wrong ``8.31 > 6.0584`` root
    bound for nvs22 — a false, super-optimal dual bound."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))
    _, elb, eub, io, isz = _extract_variable_info(m)
    tlb, tub, _, _ = tighten_root_bounds_with_fbbt(m, elb.copy(), eub.copy(), io, isz)

    for bounds in ((None), (tlb, tub)):
        res = _root_bound(m, backend=backend, bounds=bounds)
        # The fast simplex may stop at ``iteration_limit`` on the wide FBBT box
        # rather than ``optimal`` — that is fine. What is non-negotiable is the
        # soundness direction: whenever a *finite* lower bound is returned, it may
        # never exceed the true optimum. Before the cross-term conditioning guard
        # the simplex returned a wrong ``optimal`` bound of 8.31 > 6.0584 here;
        # the guard makes C4's cross-term sqrt abstain so the bound stays sound.
        if res.lower_bound is None or not math.isfinite(res.lower_bound):
            continue
        assert res.lower_bound <= optimum + 1e-3, (
            f"[{instance}/{backend}] UNSOUND root bound {res.lower_bound} > optimum {optimum}"
        )


def test_cross_term_guard_abstains_on_wide_box():
    """An *ill-conditioned* cross-term sqrt (wide box → ``|g| > 1e7``) must abstain
    rather than emit a lift the fast simplex would mis-solve.

    When the cross-term lift abstains, the only ``sqrt`` term in the objective can
    no longer be linearized, so the relaxation falls back to a feasibility
    objective and returns *no* lower bound (``lower_bound is None``) on both
    backends. That is the sound outcome — abstention drops a cut, it never
    fabricates one. The regression this locks out is the *opposite*: a finite
    ``optimal`` bound from the fast simplex that exceeds the true curve minimum
    (the #158-class wrong-"optimal" on the ill-conditioned lifted LP)."""
    # x,y in [1,4000], z in [1,4000]: g = x^2 + 2xyz + y^2 reaches ~5e10 ≫ 1e7.
    m = _cross_sqrt_model((1.0, 4000.0), (1.0, 4000.0), (1.0, 4000.0))
    fast = _root_bound(m, backend="simplex")
    ref = _root_bound(m, backend="auto")
    # True minimum of sqrt(g) over the box is at x=y=z=1: sqrt(1+2+1)=2.
    true_min = 2.0
    # Neither backend may return a finite bound that exceeds the truth. With the
    # lift abstained both return None (sound); the point is that *if* a bound were
    # returned it could not be the old super-optimal 8.31-style value.
    for tag, res in (("simplex", fast), ("auto", ref)):
        if res.lower_bound is not None and math.isfinite(res.lower_bound):
            assert res.lower_bound <= true_min + _TOL, (
                f"wide-box {tag} bound {res.lower_bound} exceeds true min {true_min} — unsound"
            )
    # And the two backends must not disagree on whether a bound exists / its value.
    if fast.lower_bound is not None and ref.lower_bound is not None:
        assert abs(fast.lower_bound - ref.lower_bound) <= 1e-3, (
            f"wide-box backend disagreement {fast.lower_bound} vs {ref.lower_bound}"
        )
    else:
        assert fast.lower_bound is None and ref.lower_bound is None, (
            f"wide-box backend disagreement on bound existence: "
            f"simplex={fast.lower_bound} auto={ref.lower_bound}"
        )
