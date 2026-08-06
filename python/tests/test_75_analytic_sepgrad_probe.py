"""The analytic separation-gradient probe must reject *uncovered atoms*, not
*unusable probe points*.

Issue #75 (Stage 1 of the JAX-removal work). ``_Builder._compiled_analytic``
(``_jax/uniform_relax.py``) computes the Kelley separation tangent ``(g(x0),
grad g(x0))`` with no JAX, via forward-mode interval AD over discopt's own
factorable IR. Before returning the pair it probes once, so that an atom the
interval-AD table does not cover is caught there rather than silently emitting no
cut deep in the tree.

That probe used ``0.5 * (lb + ub)`` and rejected the atom whenever the result was
non-finite. Two different failures were being conflated:

* **uncovered atom** — the interval-AD table has no rule (``tanh``, ``atan``,
  ``erf``, ...). This must fall back to JAX; it is the guard's whole purpose.
* **unusable probe point** — the atom is covered, but this particular point is
  outside its domain: ``0.5 * (lb + ub)`` is NaN for a free variable and +inf for
  a half-open box, and ``log`` of a box straddling zero is undefined at the
  midpoint.

Measured over 150 MINLPLib instances, the second case alone accounted for every
fallback: coverage went 96.2 % -> **100 %** (187/187 lift-node compilations) once
the probe distinguished them, with no change to the first case.

Accepting a covered-but-badly-probed atom is sound: ``_separate_convex``
(``_jax/mccormick_lp.py``) already drops any cut whose value or gradient is
non-finite — "a missing cut is always safe" — so a bad point costs a cut, never
correctness.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._jax.convexity.interval import Interval
from discopt._jax.convexity.interval_ad import interval_hessian

# Atoms the interval-AD FunctionCall table implements, and atoms it does not.
_COVERED = ["exp", "log", "sqrt", "sin", "cos", "tan"]
_UNCOVERED = ["tanh", "cosh", "sinh", "atan", "asin", "acos"]


def _point_box(model, value=1.0):
    """Degenerate (point) interval box — what the probe builds."""
    return {v: Interval(np.array([value]), np.array([value])) for v in model._variables}


def _abstained(ad) -> bool:
    """True when interval AD abstained: the densified sentinel is (-inf, +inf).

    A *covered* atom on a point box yields a degenerate interval instead, so this
    separates "no rule for this atom" from "this point is bad" — which a plain
    finiteness test cannot do.
    """
    lo = float(np.asarray(ad.value.lo).ravel()[0])
    hi = float(np.asarray(ad.value.hi).ravel()[0])
    return bool(np.isneginf(lo) and np.isposinf(hi))


@pytest.mark.unit
@pytest.mark.parametrize("fname", _UNCOVERED)
def test_uncovered_atom_still_abstains(fname):
    """The guard must survive. An atom with no interval-AD rule abstains, so
    ``_compiled_analytic`` returns None and separation falls back to JAX."""
    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    ad = interval_hessian(getattr(dm, fname)(x) * y, m, _point_box(m))
    assert _abstained(ad), (
        f"{fname} is not in the interval-AD table but did not abstain; the "
        f"uncovered-atom guard has been weakened"
    )


@pytest.mark.unit
@pytest.mark.parametrize("fname", _COVERED)
def test_covered_atom_does_not_abstain(fname):
    """A covered atom must evaluate, so the analytic path is actually taken."""
    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    ad = interval_hessian(getattr(dm, fname)(x) * y, m, _point_box(m))
    assert not _abstained(ad), f"{fname} is in the interval-AD table but abstained"


@pytest.mark.unit
@pytest.mark.parametrize(
    "lb, ub",
    [
        (0.0, np.inf),  # half-open: midpoint was +inf
        (-np.inf, 0.0),  # half-open the other way
        (-np.inf, np.inf),  # free variable: midpoint was NaN
    ],
    ids=["half-open-up", "half-open-down", "free"],
)
def test_probe_point_is_finite_for_unbounded_variables(lb, ub):
    """The probe point must be finite even when the declared box is not.

    ``0.5 * (lb + ub)`` produced NaN/inf here, and every atom on such a model was
    rejected as "uncovered" regardless of whether the table supported it — which
    is how ``dispatch`` (``x3`` free) lost its analytic separation gradients.
    """
    from discopt._jax import uniform_relax as ur

    m = Model()
    m.continuous("x", lb=lb, ub=ub)
    m.continuous("y", lb=0.5, ub=2.0)

    # The naive midpoint is non-finite on this box -- that is what used to reject
    # every atom here regardless of whether the table covered it.
    naive = np.array(
        [0.5 * (float(np.ravel(v.lb)[0]) + float(np.ravel(v.ub)[0])) for v in m._variables],
        dtype=np.float64,
    )
    assert not np.all(np.isfinite(naive)), (
        "fixture no longer reproduces the bug: the naive midpoint is finite here"
    )

    # Drive the REAL probe-point construction rather than reimplementing it, so
    # this fails if the production rule regresses.
    probe = ur._Builder._analytic_probe_point(m._variables)
    assert np.all(np.isfinite(probe)), f"probe point is not finite: {probe}"
    for value, v in zip(probe, m._variables):
        v_lo = float(np.ravel(v.lb)[0])
        v_hi = float(np.ravel(v.ub)[0])
        assert v_lo <= value <= v_hi, f"probe point {value} outside [{v_lo}, {v_hi}]"


@pytest.mark.smoke
def test_analytic_sepgrad_solves_a_lifted_model(monkeypatch):
    """End-to-end: with the analytic path on, a composite-lift model still solves.

    ``DISCOPT_ANALYTIC_SEPGRAD`` is default-OFF, so this is the only coverage that
    exercises the analytic separation gradients through a real solve — and it now
    *proves* it did (CLAUDE.md §6) instead of asserting only that some model
    solved. Two independent reasons the earlier version was vacuous, both
    measured: its model (``exp(x) + log(y) <= 12``, ``x*y >= 1``) never reached
    ``_compiled`` at all (0 calls, so nothing lifted), and even on a model that
    does lift, the dispatcher tested the graduated tape default *before* this
    flag, so ``_compiled_analytic`` was called 0 times with the flag set. The
    test passed identically with ``_compiled_analytic`` deleted.
    """
    monkeypatch.setenv("DISCOPT_ANALYTIC_SEPGRAD", "1")

    from discopt._jax import uniform_relax as ur

    calls = {"n": 0}
    _orig = ur._Builder._compiled_analytic

    def _counting(self, node, *args, **kwargs):
        calls["n"] += 1
        return _orig(self, node, *args, **kwargs)

    monkeypatch.setattr(ur._Builder, "_compiled_analytic", _counting)

    # Composite lifts (a nonlinear operand inside a nonlinear atom) are what
    # produce separation nodes; a bare `exp(x)` of a bare variable does not.
    m = Model()
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    z = m.continuous("z", lb=1.0, ub=4.0)
    m.subject_to(dm.exp(x + y) + z * z <= 60.0)
    m.subject_to((x + y) ** 2 / z >= 0.5)
    m.minimize(x + y + z)
    result = m.solve(time_limit=30.0)

    assert calls["n"] > 0, (
        "the analytic separation path was never constructed: this test does not "
        "exercise what its name and docstring claim (executed calls = 0)"
    )

    assert result.status in ("optimal", "feasible"), result.status
    assert result.objective is not None
    # Sound bound: never above the incumbent for a minimisation.
    if result.bound is not None:
        assert result.bound <= result.objective + 1e-6, (
            f"bound {result.bound!r} exceeds incumbent {result.objective!r}"
        )
