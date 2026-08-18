"""#1061 — the false-primal screen must be scale-aware, not absolute-only.

The screen's whole contract (stated in its own comment since #772) is that it
"can never flag an incumbent that is feasible within the solver's own tolerance —
only a gross violation trips it". Both call sites spelled that as ``tol=1e-3`` and
inherited ``rtol``'s *acceptance* default of 1e-9, which loosens only the absolute
leg. On a row whose terms are large the screen therefore collapses to an
absolute-only test and withholds converged points.

Measured on ``nvs05`` before this fix: row 2 has scale ``sum_j |J_2j|*|x_j|`` =
73,674; a point whose RELATIVE residual there is 2.8e-8 (converged by IPOPT's own
default tolerance, and within 1e-7 per component of the point the same solver
certifies optimal) carries a 2.06e-3 absolute residual and was reported as
``status="error"``, objective withheld. The default path sits at 1.34e-4 on that
same row — a factor of 7.5 from tripping.

These tests fix the boundary in both directions: a converged point on a
large-magnitude row is kept, and a violation that is large *relative to the row's
own scale* is still refused, so the relative leg is not a blank cheque.
"""

from __future__ import annotations

import pathlib

import discopt.modeling as dm
import discopt.solver as solver_mod
import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt.modeling.core import SolveResult

# x**3 - y == 0, cubic so the model is genuinely nonlinear (a quadratic row routes
# to the fast LP/MILP/QP family, which skips the verification snapshot entirely and
# would make every assertion below vacuous).
_X = 30.0
_Y = _X**3  # 27000
# scale = |d/dx|*|x| + |d/dy|*|y| = 3*30^2*30 + 1*27000 = 108000
_ROW_SCALE = 3 * _X**2 * _X + _Y


def _cubic_model():
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=100.0)
    y = m.continuous("y", lb=0.0, ub=1.0e6)
    m.subject_to(x * x * x - y == 0.0)
    m.minimize(y)
    return m


def _solve_with_injected_point(monkeypatch, x_val: float, y_val: float):
    """Return ``(result, screen_calls)`` for an injected incumbent.

    ``screen_calls`` is the CLAUDE.md §6 counter: without it, "the guard accepted
    the point" and "the guard never ran" are indistinguishable, since both leave
    ``incumbent_verification_failed`` False.
    """
    calls = {"n": 0}
    real = ph.passes_false_primal_screen

    def _counting(evaluator, x):
        calls["n"] += 1
        return real(evaluator, x)

    monkeypatch.setattr(ph, "passes_false_primal_screen", _counting)

    def _fake_solve_model(model, **kwargs):
        return SolveResult(
            status="optimal",
            objective=y_val,
            bound=y_val,
            gap=0.0,
            x={"x": np.array(x_val), "y": np.array(y_val)},
            gap_certified=True,
        )

    monkeypatch.setattr(solver_mod, "solve_model", _fake_solve_model)
    m = _cubic_model()
    return m.solve(time_limit=5), calls


def test_converged_point_on_a_large_row_is_not_a_false_primal(monkeypatch):
    """The nvs05 signature: absolute residual over 1e-3, relative residual ~1e-8.

    Fails before the fix (the screen was effectively absolute-only and withheld
    this point); passes after.
    """
    resid = 2.0e-3
    r, calls = _solve_with_injected_point(monkeypatch, _X, _Y + resid)

    assert calls["n"] > 0, "the screen never ran — this test asserted nothing"
    rel = resid / _ROW_SCALE
    assert rel < 1e-7, f"fixture is not the intended regime (rel={rel:.2e})"
    assert resid > ph.FALSE_PRIMAL_ATOL, "fixture must exceed the absolute leg"

    assert r.incumbent_verification_failed is False, (
        f"a point with relative residual {rel:.2e} on a row of scale {_ROW_SCALE:g} "
        f"was reported as a false primal — the screen is scale-blind"
    )
    assert r.x is not None and r.objective is not None
    assert r.status == "optimal"


def test_a_violation_large_relative_to_the_row_is_still_refused(monkeypatch):
    """The relative leg is not a blank cheque: a residual that is large *in the
    row's own units* still trips the screen, on the very same large-scale row."""
    resid = 10.0  # rel = 9.3e-5, ~93x the relative leg
    r, calls = _solve_with_injected_point(monkeypatch, _X, _Y + resid)

    assert calls["n"] > 0, "the screen never ran — this test asserted nothing"
    assert resid / _ROW_SCALE > ph.FALSE_PRIMAL_RTOL, "fixture below the relative leg"

    assert r.incumbent_verification_failed is True
    assert r.x is None and r.objective is None
    assert r.gap_certified is False
    assert r.status == "error"


def test_a_gross_violation_is_still_refused(monkeypatch):
    """The #770 failure mode (violations 0.4–17.6) is untouched."""
    r, calls = _solve_with_injected_point(monkeypatch, _X, 0.0)  # residual 27000

    assert calls["n"] > 0, "the screen never ran — this test asserted nothing"
    assert r.incumbent_verification_failed is True
    assert r.x is None and r.objective is None


def test_a_well_scaled_row_keeps_the_1e_3_behaviour():
    """On a row whose terms are O(1) the relative leg contributes ~1e-6, so the
    screen still behaves as the plain absolute 1e-3 test it has always been.

    This is what makes the change safe: it only ever differs where the absolute
    test was meaningless.
    """
    from discopt._relax.nlp_evaluator import cached_evaluator

    m = dm.Model()
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(x * x * x + y * y * y <= 1.0)
    m.minimize(x + y)
    ev = cached_evaluator(m)

    checks = 0
    # (point, expected) — the boundary sits at 1e-3 + 1e-6*scale, and with terms of
    # size O(1) that second term cannot move the verdict at these magnitudes.
    for pt, expected in (
        (np.array([0.5, 0.5]), True),  # 0.25 <= 1, comfortably feasible
        (np.array([1.0, 1.0]), False),  # 2.0 vs 1.0 — a gross violation
    ):
        assert ph.passes_false_primal_screen(ev, pt) is expected, f"{pt} -> {expected}"
        checks += 1
    assert checks == 2


@pytest.mark.parametrize(
    "rel_path",
    ["python/discopt/modeling/core.py", "python/discopt/solver.py"],
)
def test_neither_call_site_hardcodes_its_own_screen_tolerance(rel_path):
    """The two screens must not drift apart.

    ``solver.py`` already claimed to use "the SAME loose check" as ``core.py``, and
    the only thing making that true was the same literal typed into both. Keeping
    the tolerance in one place is the fix; this test keeps it there.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / rel_path).read_text()
    assert "passes_false_primal_screen" in src, f"{rel_path} no longer uses the shared screen"

    # Comments are allowed to name the old literal (they explain why it is gone);
    # executable code is not.
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "tol=1e-3" not in code, (
        f"{rel_path} hardcodes a false-primal tolerance again; use "
        f"passes_false_primal_screen so the two sites cannot diverge"
    )


def test_the_old_absolute_only_tolerance_is_what_rejected_the_point():
    """Pin the mechanism directly, in one run, without reverting anything.

    The other tests here fail on the pre-fix tree because the shared screen does
    not exist there, which proves they are new but not *why* they are needed. This
    one compares the two tolerance models side by side on the same evaluator and
    the same point: the old ``tol=1e-3, rtol=1e-9`` pair rejects a point whose
    relative residual is ~1e-8, and the new pair accepts it. That difference is the
    entire bug.
    """
    from discopt._relax.nlp_evaluator import cached_evaluator

    m = _cubic_model()
    ev = cached_evaluator(m)
    resid = 2.0e-3
    pt = np.array([_X, _Y + resid])

    old_verdict = ph._check_constraint_feasibility(ev, pt, tol=1e-3, rtol=1e-9)
    new_verdict = ph.passes_false_primal_screen(ev, pt)

    assert old_verdict is False, (
        "fixture no longer reproduces the bug: the old absolute-only screen "
        "accepts this point, so there is nothing to fix"
    )
    assert new_verdict is True, (
        f"the scale-aware screen still rejects a point whose relative residual is "
        f"{resid / _ROW_SCALE:.2e} on a row of scale {_ROW_SCALE:g}"
    )
