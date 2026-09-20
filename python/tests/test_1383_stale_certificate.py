"""#1383 — a certificate may not outlive the pair it was earned on.

The defect
----------

A tree converges honestly: incumbent ≈ bound, so ``stats["gap"] = 0.0`` and the
certificate is earned. The assembly that follows then *changes the objective* —
here the post-solve incumbent refinement (re-solve the NLP with integers fixed)
— and nothing re-tests the certificate. ``gap`` and ``bound`` are left describing
the point that was replaced.

Measured on a 5-variable random MINLP with its one equality split into two
inequalities, asking for ``gap_tolerance=1e-9, abs_gap_tolerance=1e-10``::

    status="optimal"  objective=-0.7432619320556588  bound=-0.7433375500435884
    gap=0.0           gap_certified=True

``objective - bound`` is 7.56e-5 — five orders past either requested tolerance —
and an independent scipy multistart (2400 local solves) puts the true optimum at
-0.7433375895, so the reported incumbent was also 7.6e-5 *worse than optimal*
while claiming optimality.

The two layers, and why both
----------------------------

1. **The cause.** The refinement's "objectives match" branch is a 1e-4 *relative*
   window that then decides on constraint excess alone, so a point worse by 7.6e-5
   was adopted over an incumbent that already cleared the exit gate. That branch
   exists to repair feasibility; with nothing to repair, adopting a worse point
   only discards the optimum just proved.

2. **The class.** :func:`_withhold_stale_certificate` re-tests the FINAL
   ``(objective, bound)`` pair at every B&B assembly site, with the same predicate
   that stops the search. Any future stage that moves the objective after the gap
   was computed is caught, not just this one.

These tests pin both layers separately, so removing either fails a test: the
end-to-end cases pin (1) by requiring the right answer, and the unit tests pin (2)
by requiring the withdrawal on a pair constructed here.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from scipy.optimize import minimize

# ── the model, and an oracle that never touches discopt ────────────────────

#: Independent reference optimum: scipy SLSQP, 400 multistart points per integer
#: assignment over the 6 assignments, computed by :func:`_scipy_oracle` below.
#: Pinned so the suite does not pay for 2400 local solves on every run; the
#: ``slow``-marked test recomputes it and fails if it has drifted.
ORACLE = -0.7433375895357062

_LIN = np.array([-0.800850801180288, 0.2887547170645248, -0.5891242470598537])
_QUAD = np.array([0.38469266956128556, 0.22416123434946128, 1.03681875542562])
_BIL = np.array([0.29204644088885945, 0.14045115898934])
_ZLIN = np.array([-0.6821310405105137, -0.3791274716418804])
_W = np.array([-0.32445226913516456, -0.17128539967518597, -0.8789600292182848])
_RHS = 0.4168305693695522
_V = np.array([-0.867344772890708, -0.6318557919811769, 0.38104599707818104])
_ERHS = -0.00013746164389250026
_LO = np.array([-0.14072798874084036, -1.1837511661047286, -2.505494657123682])
_HI = np.array([2.04368239910949, 2.3372628283609815, 1.0120186331755527])
_IUB = (2, 1)


def _f(x, z):
    """The objective, in plain numpy."""
    val = float(_LIN @ x + _QUAD @ (x * x))
    for k in range(2):
        val += _BIL[k] * x[k] * x[k + 1]
    val += float(_ZLIN @ z) + 0.3 * float(z @ z)
    val += 0.5 * float(np.exp(0.25 * x[0]))
    return val


def _build(split: bool):
    """The model; ``split`` replaces the equality with two inequalities.

    The two forms describe the identical feasible set, so every certified answer
    must agree — that equivalence is the whole probe.
    """
    m = dm.Model("c1383")
    xs = [m.continuous(f"c{k}", lb=float(_LO[k]), ub=float(_HI[k])) for k in range(3)]
    zs = [m.integer(f"i{3 + k}", lb=0, ub=_IUB[k]) for k in range(2)]

    obj = 0.0
    for k in range(3):
        obj = obj + float(_LIN[k]) * xs[k] + float(_QUAD[k]) * xs[k] * xs[k]
    for k in range(2):
        obj = obj + float(_BIL[k]) * xs[k] * xs[k + 1]
    for k in range(2):
        obj = obj + float(_ZLIN[k]) * zs[k] + 0.3 * zs[k] * zs[k]
    obj = obj + 0.5 * dm.exp(0.25 * xs[0])
    m.minimize(obj)

    ineq = sum(float(_W[k]) * xs[k] for k in range(3)) + zs[0]
    m.subject_to(ineq <= _RHS, name="ineq")

    ebody = sum(float(_V[k]) * xs[k] for k in range(3))
    if split:
        m.subject_to(ebody <= _ERHS, name="eq_le")
        m.subject_to(ebody >= _ERHS, name="eq_ge")
    else:
        m.subject_to(ebody == _ERHS, name="eq")
    return m


def _scipy_oracle(starts: int = 400):
    """The optimum by enumeration + multistart SLSQP. No discopt involved."""
    rng = np.random.default_rng(12345)
    best = None
    solved = 0
    for z in itertools.product(*[range(u + 1) for u in _IUB]):
        zv = np.array(z, dtype=float)
        cons = [
            {"type": "ineq", "fun": lambda x, zv=zv: _RHS - (_W @ x + zv[0])},
            {"type": "eq", "fun": lambda x: _V @ x - _ERHS},
        ]
        for _ in range(starts):
            x0 = _LO + (_HI - _LO) * rng.random(3)
            r = minimize(
                lambda x, zv=zv: _f(x, zv),
                x0,
                method="SLSQP",
                bounds=list(zip(_LO, _HI)),
                constraints=cons,
                options={"maxiter": 400, "ftol": 1e-14},
            )
            solved += 1
            if not r.success:
                continue
            x = np.clip(r.x, _LO, _HI)
            if _W @ x + zv[0] > _RHS + 1e-9 or abs(_V @ x - _ERHS) > 1e-9:
                continue
            val = _f(x, zv)
            if best is None or val < best:
                best = val
    assert solved > 0, "the oracle ran no local solves"
    return best


# ── layer 1: the answer itself ─────────────────────────────────────────────


@pytest.mark.parametrize("split", [False, True])
def test_neither_form_certifies_above_the_oracle(split):
    """The headline invariant: a certificate may not sit above the true optimum."""
    r = _build(split).solve(time_limit=120, gap_tolerance=1e-9, abs_gap_tolerance=1e-10)
    if r.objective is None:
        pytest.skip(f"no incumbent (status={r.status})")
    assert r.objective >= ORACLE - 1e-6, (
        f"split={split}: incumbent {r.objective} is BELOW the independent oracle "
        f"{ORACLE} — no feasible point of this model attains it"
    )
    if r.status == "optimal":
        assert r.objective == pytest.approx(ORACLE, abs=1e-6), (
            f"split={split}: certified optimal at {r.objective}, oracle says {ORACLE} "
            f"(off by {r.objective - ORACLE:.3e})"
        )


def test_splitting_the_equality_does_not_move_the_certified_optimum():
    """`a == b` and (`a <= b`, `a >= b`) are the same feasible set."""
    eq = _build(False).solve(time_limit=120, gap_tolerance=1e-9, abs_gap_tolerance=1e-10)
    sp = _build(True).solve(time_limit=120, gap_tolerance=1e-9, abs_gap_tolerance=1e-10)
    if eq.status != "optimal" or sp.status != "optimal":
        pytest.skip(f"not both certified (eq={eq.status}, split={sp.status})")
    assert eq.objective == pytest.approx(sp.objective, abs=1e-6), (
        f"the equality form certified {eq.objective} and the split form "
        f"{sp.objective}; the two models have the identical feasible set"
    )


# ── the self-consistency contract, on any route ────────────────────────────


@pytest.mark.parametrize("split", [False, True])
def test_a_certified_result_is_self_consistent(split):
    """``gap_certified`` may not stand beside a pair that does not close the gap.

    This is the invariant the defect broke in the most visible way: ``gap=0.0``
    and ``gap_certified=True`` published beside an ``objective``/``bound`` pair
    1.0e-4 apart, at a requested tolerance of 1e-9.
    """
    rel, abs_ = 1e-9, 1e-10
    r = _build(split).solve(time_limit=120, gap_tolerance=rel, abs_gap_tolerance=abs_)
    if not getattr(r, "gap_certified", False):
        return  # an uncertified result makes no claim to check
    assert r.objective is not None and r.bound is not None
    spread = abs(r.objective - r.bound)
    allowed = max(abs_, rel * max(abs(r.objective), abs(r.bound)))
    assert spread <= allowed + 1e-9, (
        f"split={split}: gap_certified=True with objective={r.objective} and "
        f"bound={r.bound} — a spread of {spread:.3e} against an allowance of "
        f"{allowed:.3e} (gap field reads {r.gap})"
    )


# ── layer 2: the guard, exercised directly ─────────────────────────────────


def _guard():
    """The guard, imported lazily.

    Kept out of module scope on purpose: this file must still COLLECT against a
    tree that does not have the fix, so that the end-to-end tests above fail on
    the behaviour they are about rather than the whole module erroring on an
    import (which reads as "the test is broken", not "the solver is").
    """
    from discopt.solver import _withhold_stale_certificate

    return _withhold_stale_certificate


def test_the_guard_withdraws_a_certificate_the_pair_does_not_support():
    """The exact numbers from the incident, fed to the guard on their own."""
    status, gap, certified = _guard()(
        "optimal",
        -0.7432619320556588,
        -0.7433375500435884,
        0.0,  # the stale gap, describing the incumbent that was replaced
        True,
        False,  # minimize
        1e-9,
        1e-10,
        "unit",
    )
    assert certified is False, "the guard kept a certificate the pair contradicts"
    assert status == "feasible", f"status stayed {status!r}"
    assert gap == pytest.approx(7.561798792954377e-05, rel=1e-6), (
        f"the stale gap=0.0 was not replaced with the honest one (got {gap})"
    )


def test_the_guard_keeps_a_certificate_the_pair_does_support():
    """It must not cost a single legitimate certificate."""
    status, gap, certified = _guard()(
        "optimal",
        -0.7433375500435884,
        -0.7433375500435884,
        0.0,
        True,
        False,
        1e-9,
        1e-10,
        "unit",
    )
    assert certified is True and status == "optimal" and gap == 0.0

    # ...and a pair converged on the RELATIVE arm at a loose tolerance, which
    # reports gap=0.0 by design even though the spread is nonzero.
    status, gap, certified = _guard()(
        "optimal",
        1.0,
        1.0 - 5e-5,
        0.0,
        True,
        False,
        1e-4,
        1e-10,
        "unit",
    )
    assert certified is True, "a legitimate relative-arm convergence was withdrawn"
    assert status == "optimal" and gap == 0.0


def test_the_guard_is_downgrade_only():
    """It never manufactures a certificate, whatever it is handed."""
    for status in ("feasible", "time_limit", "node_limit"):
        out_status, out_gap, out_cert = _guard()(
            status,
            1.0,
            1.0,
            0.5,
            False,
            False,
            1e-9,
            1e-10,
            "unit",
        )
        assert out_cert is False, f"{status}: uncertified input came back certified"
        assert out_status == status and out_gap == 0.5, "an uncertified result was altered"


@pytest.mark.parametrize(
    "obj,bound,is_max",
    [
        (None, -1.0, False),
        (-1.0, None, False),
        (float("nan"), -1.0, False),
        (-1.0, float("inf"), False),
    ],
)
def test_the_guard_passes_through_an_unusable_pair(obj, bound, is_max):
    """No pair to test means no judgement — never an invented withdrawal."""
    status, gap, certified = _guard()(
        "optimal",
        obj,
        bound,
        0.0,
        True,
        is_max,
        1e-9,
        1e-10,
        "unit",
    )
    assert (status, gap, certified) == ("optimal", 0.0, True)


def test_the_guard_handles_the_maximize_sense():
    """A MAXIMIZE bound is an UPPER bound; the sense must not invert the test."""
    # A genuine maximize certificate: bound just above the incumbent.
    _s, _g, cert = _guard()(
        "optimal",
        10.0,
        10.0 + 1e-12,
        0.0,
        True,
        True,
        1e-9,
        1e-10,
        "unit",
    )
    assert cert is True, "a valid maximize certificate was withdrawn"
    # A stale one: the bound is 1e-3 above the incumbent, far past 1e-9.
    status, gap, cert = _guard()(
        "optimal",
        10.0,
        10.001,
        0.0,
        True,
        True,
        1e-9,
        1e-10,
        "unit",
    )
    assert cert is False and status == "feasible", "a stale maximize certificate stood"
    assert gap == pytest.approx(1e-4, rel=1e-6)


# ── the oracle itself, so the pinned constant cannot rot ───────────────────


@pytest.mark.slow
def test_the_pinned_oracle_still_matches_a_fresh_multistart():
    """Recompute the reference; ``ORACLE`` is evidence, not a magic number."""
    fresh = _scipy_oracle(starts=400)
    assert fresh == pytest.approx(ORACLE, abs=1e-9), (
        f"the pinned oracle {ORACLE} no longer matches a fresh scipy multistart {fresh}"
    )
