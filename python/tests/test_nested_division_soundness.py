"""Adversarial soundness for the nested-division lift (issue #154, increment 3).

The lifted reciprocal envelope (#157) only claims ``c / g`` with a *constant*
numerator. A division whose numerator is *non-constant* — nvs05/nvs22 constraint
``C2 = (0.5*x3)/x6`` — instead hit ``Cannot linearize non-constant division`` and
the whole constraint was dropped. Increment 3 lifts it as the factorable product

    num / g  =  n · (ncoeff / g)        with   num = ncoeff · n,  ncoeff > 0

building a reciprocal aux ``r = ncoeff/g`` (the scalar folded into the affine
argument ``g/ncoeff``) and a McCormick product ``P = n · r``, then substituting
the division node with ``P``.

Two invariants are locked here, both the project's non-negotiable one — a valid
lower bound never exceeds the true optimum, and the relaxation never excludes a
feasible point:

1. **Envelope encloses the curve.** On well-conditioned boxes the underestimator
   never rises above ``num/g`` and the overestimator never drops below it (probed
   by pinning to interior grid points, where the McCormick product + reciprocal
   envelope collapse to the exact value).

2. **Ill-conditioned LP → wrong "optimal".** When the product-aux bounds reach the
   magnitude where the lifted LP is ill-conditioned (the #158 / increment-2
   hazard), the fast simplex can return a wrong "optimal" that *exceeds* the true
   optimum. The conditioning guard (``_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE``) makes
   the lift abstain there, so the relaxation stays sound on every backend. The two
   MINLPLib cases are the regression lock: their root bound must never exceed the
   optimum on either backend.

It also pins the *engagement* the increment delivers: nvs22's ``(0.5*x3)/x6`` is
no longer dropped from the relaxation.
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


def _div_model(xb, yb, *, over=False):
    """``(0.5*y)/x`` over the given boxes — a non-constant (monomial) numerator
    over a single-variable denominator, the shape increment 3 unlocks.

    ``over=False`` minimizes the division (exercises the *underestimator*).
    ``over=True`` minimizes ``0 - (0.5*y)/x`` — the leading minus stays *outside*
    the division node (so the numerator coefficient stays positive and the lift
    still fires), and minimizing the negation exercises the *overestimator*.
    """
    m = dm.Model()
    x = m.continuous("x", lb=xb[0], ub=xb[1])
    y = m.continuous("y", lb=yb[0], ub=yb[1])
    expr = (0.5 * y) / x
    m.minimize(0.0 - expr if over else expr)
    return m


def _root_bound(model, backend="simplex", pin=None, bounds=None):
    relaxer = MccormickLPRelaxer(model)
    relaxer._backend = backend
    if bounds is None:
        lb, ub = flat_variable_bounds(model)
    else:
        lb, ub = bounds
    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    if pin is not None:
        for i, v in pin.items():
            lb[i] = v
            ub[i] = v
    return relaxer.solve_at_node(lb, ub)


# Well-conditioned boxes: the product-aux bounds stay well below the conditioning
# guard, so the lift engages and the fast simplex is reliable. ``x`` is kept
# strictly positive (reciprocal needs a positive denominator).
_WELL_CONDITIONED = [
    ((1.0, 3.0), (1.0, 3.0)),
    ((0.5, 2.0), (1.0, 4.0)),
    ((2.0, 5.0), (0.5, 3.0)),
]


@pytest.mark.parametrize("xb, yb", _WELL_CONDITIONED)
def test_nested_division_lift_engages(xb, yb):
    """On a well-conditioned box the nested-division lift creates aux columns (a
    reciprocal aux + a McCormick product) — it fires rather than abstaining."""
    m = _div_model(xb, yb)
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    milp, _ = build_milp_relaxation(
        relaxer._model,
        relaxer._terms,
        relaxer._disc,
        bound_override=(np.asarray(lb), np.asarray(ub)),
        superposition=relaxer._superposition,
    )
    # 2 original variables; the lift adds a reciprocal aux and a product aux.
    assert len(milp._c) > 2, "nested-division lift did not engage (no aux columns)"


@pytest.mark.parametrize("xb, yb", _WELL_CONDITIONED)
def test_nested_division_encloses_curve(xb, yb):
    """At every interior grid point the relaxation brackets ``(0.5*y)/x`` from both
    sides — the nested-division lift never excludes the true curve."""
    xs = [xb[0], 0.5 * (xb[0] + xb[1]), xb[1]]
    ys = [yb[0], 0.5 * (yb[0] + yb[1]), yb[1]]
    for xv in xs:
        for yv in ys:
            true = (0.5 * yv) / xv
            pin = {0: xv, 1: yv}
            lo = _root_bound(_div_model(xb, yb), pin=pin)
            hi = _root_bound(_div_model(xb, yb, over=True), pin=pin)
            assert lo.status == "optimal" and lo.lower_bound is not None
            assert hi.status == "optimal" and hi.lower_bound is not None
            # underestimator never rises above the curve …
            assert lo.lower_bound <= true + _TOL, (
                f"nested-div under-cut excludes curve at ({xv},{yv}): {lo.lower_bound} > {true}"
            )
            # … overestimator (negated objective) never drops below it.
            assert -hi.lower_bound >= true - _TOL, (
                f"nested-div over-cut excludes curve at ({xv},{yv}): {-hi.lower_bound} < {true}"
            )


@pytest.mark.parametrize("xb, yb", _WELL_CONDITIONED)
def test_nested_division_backends_agree(xb, yb):
    """In the well-conditioned regime the fast simplex and HiGHS must agree — a
    disagreement would mean the lifted LP is already ill-conditioned and the fast
    bound is untrustworthy."""
    m = _div_model(xb, yb)
    fast = _root_bound(m, backend="simplex")
    ref = _root_bound(m, backend="auto")
    assert fast.status == "optimal" and ref.status == "optimal"
    assert abs(fast.lower_bound - ref.lower_bound) <= 1e-4, (
        f"backend disagreement {fast.lower_bound} vs {ref.lower_bound} — fast bound unreliable"
    )


def test_nested_division_abstains_on_negative_numerator():
    """A *negative* numerator coefficient is out of scope — folding it into the
    reciprocal argument would flip the denominator interval negative (the convex
    reciprocal envelope needs a positive denominator). The lift must abstain, which
    leaves the constraint un-lifted (sound: it only enlarges the relaxation)."""
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    m.minimize((-0.5 * y) / x)
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    milp, _ = build_milp_relaxation(
        relaxer._model,
        relaxer._terms,
        relaxer._disc,
        bound_override=(np.asarray(lb), np.asarray(ub)),
        superposition=relaxer._superposition,
    )
    # No lift → no aux columns beyond the 2 originals.
    assert len(milp._c) == 2, "negative-numerator division should not lift"


# nvs05/nvs22: constraint C2 = (0.5*x3)/x6 now lifts; (instance, optimum).
_MINLPLIB = [("nvs05", 5.47093), ("nvs22", 6.0584)]


@pytest.mark.parametrize("instance, optimum", _MINLPLIB)
@pytest.mark.parametrize("backend", ["simplex", "auto"])
def test_nested_division_keeps_minlplib_sound(instance, optimum, backend):
    """With C2's nested division now lifted, nvs05/nvs22 root bounds must stay
    sound on *both* backends, under default and FBBT-tightened bounds. The
    non-negotiable invariant: a valid lower bound never exceeds the optimum."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))
    _, elb, eub, io, isz = _extract_variable_info(m)
    tlb, tub, _, _ = tighten_root_bounds_with_fbbt(m, elb.copy(), eub.copy(), io, isz)

    for bounds in ((None), (tlb, tub)):
        res = _root_bound(m, backend=backend, bounds=bounds)
        # A non-optimal status (e.g. iteration_limit on the wide box) is fine; what
        # is non-negotiable is that any *finite* bound returned is sound.
        if res.lower_bound is None or not math.isfinite(res.lower_bound):
            continue
        assert res.lower_bound <= optimum + 1e-3, (
            f"[{instance}/{backend}] UNSOUND root bound {res.lower_bound} > optimum {optimum}"
        )


def test_nested_division_engages_on_nvs22(caplog):
    """The increment's headline: nvs22's ``(0.5*x3)/x6`` is no longer dropped from
    the relaxation. Before increment 3 it appeared in the AMP omit warnings as a
    non-constant division; now it is linearized."""
    import logging

    nl = _DATA / "nvs22.nl"
    m = dm.from_nl(str(nl))
    _, elb, eub, io, isz = _extract_variable_info(m)
    tlb, tub, _, _ = tighten_root_bounds_with_fbbt(m, elb.copy(), eub.copy(), io, isz)
    with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
        _root_bound(m, backend="auto", bounds=(tlb, tub))
    omitted = "\n".join(
        r.getMessage() for r in caplog.records if "omitting constraint" in r.getMessage()
    )
    assert "(0.5 * x3) / x6" not in omitted, (
        "nvs22 C2 (0.5*x3)/x6 is still omitted — the nested-division lift did not engage"
    )


def test_nested_division_guard_keeps_wide_box_sound():
    """An *ill-conditioned* nested division — a large numerator over a small
    denominator pushes the product-aux bounds past the magnitude guard — must not
    yield a finite bound that exceeds the truth on the fast simplex. The guard
    makes the lift abstain (no bound), which is the sound outcome; the regression
    this locks out is a wrong super-optimal "optimal" from the fast backend."""
    # y in [1, 1e8], x in [0.5, 1.0]: product n·r reaches ~1e8 ≫ 1e7 → abstain.
    m = _div_model((0.5, 1.0), (1.0, 1.0e8))
    fast = _root_bound(m, backend="simplex")
    ref = _root_bound(m, backend="auto")
    # True minimum of (0.5*y)/x over the box is at y=1, x=1: 0.5.
    true_min = 0.5
    for tag, res in (("simplex", fast), ("auto", ref)):
        if res.lower_bound is not None and math.isfinite(res.lower_bound):
            assert res.lower_bound <= true_min + _TOL, (
                f"wide-box {tag} bound {res.lower_bound} exceeds true min {true_min} — unsound"
            )
    if fast.lower_bound is not None and ref.lower_bound is not None:
        assert abs(fast.lower_bound - ref.lower_bound) <= 1e-3, (
            f"wide-box backend disagreement {fast.lower_bound} vs {ref.lower_bound}"
        )
    else:
        assert fast.lower_bound is None and ref.lower_bound is None, (
            f"wide-box backend disagreement on bound existence: "
            f"simplex={fast.lower_bound} auto={ref.lower_bound}"
        )
