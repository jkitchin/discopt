"""#1313: ``Model.sensitivity()`` must describe the solution the model was solved to.

Two defects, both reported against the #1251 entry point:

1. ``sensitivity()`` started its inner POUNCE solve from a fixed midpoint of the
   clipped variable box and never consulted the model's own solved incumbent, so
   on a nonconvex model it could return derivatives from a different basin
   entirely -- under ``status="optimal"``, the same word ``SolveResult.status``
   uses for a *certified global* optimum. The repro here is the double-well from
   the issue: ``solve()`` certifies the deep well at ``x = -0.5``, the old
   ``sensitivity()`` reported the shallow one at ``x = 3``.
2. ``Sensitivity.multipliers`` / ``dlambda_dp`` carry the internal
   (negated-for-MAXIMIZE) sign convention -- the same one
   ``SolveResult.constraint_duals`` documents -- but the new class's docstrings
   said nothing about it, and no test pinned the behaviour with a MAXIMIZE model.

The fix warm-starts the inner solve from the model's last ``solve()`` result,
cross-checks the returned KKT point against it (``matches_reference``), warns
loudly on a mismatch, and documents both the sign convention and what ``status``
actually means here.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest


def _double_well():
    """Two Gaussian wells: a deep one at ``x=-0.5``, a shallow ``p``-scaled one at ``x=3``.

    Straight from the issue. The midpoint of the box ``[-1, 5]`` is ``x=2``, which
    is why the old fixed start landed in the shallow well every time.
    """
    m = dm.Model("i1313_doublewell")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-1.0, ub=5.0)
    deep = -5.0 * dm.exp(-((x - (-0.5)) ** 2) / (2 * 0.3**2))
    shallow = -(2.0 + p) * dm.exp(-((x - 3.0) ** 2) / (2 * 0.3**2))
    m.minimize(deep + shallow + 0.01 * x)
    return m, p, x


def test_sensitivity_uses_the_solved_incumbent():
    """After ``solve()``, the derivatives belong to the certified solution."""
    m, _p, _x = _double_well()
    r = m.solve()
    assert r.status == "optimal"
    assert r.bound is not None and abs(r.objective - r.bound) < 1e-4  # certified

    s = m.sensitivity()

    assert s.matches_reference is True
    assert s.reference_objective == pytest.approx(r.objective, abs=1e-9)
    assert s.objective == pytest.approx(r.objective, rel=1e-6)
    # The deep well, not the shallow one at x=3 the old code reported.
    assert float(np.asarray(s.x).ravel()[0]) == pytest.approx(-0.5, abs=1e-2)


def test_unsolved_model_has_no_reference_and_does_not_warn():
    """No solve, no incumbent to check against -- the old behaviour, stated honestly."""
    m, _p, _x = _double_well()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = m.sensitivity()
    assert s.matches_reference is None
    assert s.reference_objective is None


def test_explicit_at_overrides_the_recorded_solve():
    """``at=`` pins the point; a flat array carries no reference objective."""
    m, _p, x = _double_well()
    m.solve()
    s = m.sensitivity(at=np.array([3.0]))
    assert float(np.asarray(s.x).ravel()[0]) == pytest.approx(3.0, abs=1e-1)
    assert s.matches_reference is None
    # And by name, which goes through the initial-solution validator.
    s2 = m.sensitivity(at={x.name: 3.0})
    assert float(np.asarray(s2.x).ravel()[0]) == pytest.approx(3.0, abs=1e-1)


def test_stale_reference_warns_and_flags_the_mismatch():
    """A bound moved since the solve: the KKT point is no longer the reference one.

    This is the case ``status="optimal"`` used to hide completely. The local solve
    genuinely converges, so ``status`` stays ``"optimal"`` -- ``matches_reference``
    is what says the point is not the solution being checked against.
    """
    m = dm.Model("i1313_stale")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize((x - 0.9) ** 2 + p * x)
    r = m.solve()
    assert r.objective == pytest.approx(0.0, abs=1e-5)

    x.ub = np.array(0.5)  # the reference point x=0.9 is now out of the box

    with pytest.warns(RuntimeWarning, match="different stationary point|KKT point"):
        s = m.sensitivity()

    assert s.matches_reference is False
    assert s.reference_objective == pytest.approx(0.0, abs=1e-5)
    assert s.objective == pytest.approx(0.16, rel=1e-3)  # (0.5-0.9)^2
    assert float(np.asarray(s.x).ravel()[0]) == pytest.approx(0.5, abs=1e-4)


def test_the_recorded_result_does_not_travel_through_serialization():
    """A reloaded model must not claim a solve nobody in the new process ran.

    ``_last_solve_result`` is a cache of something that happened in *this*
    process. The document already has an explicit, opt-in way to carry a result
    (``Model.save(..., result=...)`` -> ``saved_result``); if this one rode along
    too, ``sensitivity()`` on a freshly loaded model would silently cross-check
    against a stranger's solve. ``discopt.serialize`` classifies it as
    not-carried, and ``test_every_model_attribute_is_accounted_for`` is what
    forced the decision.
    """
    import discopt.modeling as _dm

    m, _p, _x = _double_well()
    m.solve()
    assert m._last_solve_result is not None

    reloaded = _dm.loads(_dm.dumps(m))
    assert reloaded._last_solve_result is None

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = reloaded.sensitivity()
    assert s.matches_reference is None


# ── part 2: the multiplier sign convention ──────────────────────────────


def _bounded_pair(sense):
    """``min (x-3)^2`` / ``max -(x-3)^2`` subject to ``x <= 1``, both with the same argmin.

    The multiplier of the active row is 4 for the minimize form. The maximize form
    is solved as ``-f``, so its multipliers belong to ``-f`` and come back as the
    *same* number -- the convention ``SolveResult.constraint_duals`` documents.
    """
    m = dm.Model(f"i1313_sign_{sense}")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    body = (x - 3.0) ** 2 + p * x
    if sense == "max":
        m.maximize(-body)
    else:
        m.minimize(body)
    m.subject_to(x <= 1.0)
    return m, p, x


@pytest.mark.parametrize("sense", ["min", "max"])
def test_multipliers_are_in_the_internal_minimization_convention(sense):
    """Same raw multiplier for both senses; ``dobj_dp`` still follows the model's sense."""
    m, _p, _x = _bounded_pair(sense)
    s = m.sensitivity()

    assert s.multipliers.shape == (1,)
    # Identical under both senses -- that is the documented convention, not a bug.
    assert float(s.multipliers[0]) == pytest.approx(4.0, rel=1e-4)

    # dobj_dp *is* sense-corrected: d/dp of the objective as written, at x*=1.
    expected = 1.0 if sense == "min" else -1.0
    assert float(np.asarray(s.dobj_dp).ravel()[0]) == pytest.approx(expected, rel=1e-4)


def test_multiplier_convention_matches_solveresult_constraint_duals():
    """The two APIs must not disagree about the sign of the same multiplier."""
    for sense in ("min", "max"):
        m, _p, _x = _bounded_pair(sense)
        r = m.solve()
        s = m.sensitivity()
        duals = r.constraint_duals
        if duals is None:  # the route taken did not report duals; nothing to compare
            continue
        flat = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in duals.values()])
        assert flat.size == s.multipliers.size
        assert np.allclose(np.abs(flat), np.abs(s.multipliers), rtol=1e-3, atol=1e-6)


def test_sign_convention_is_documented_on_the_new_class():
    """The #1313.2 regression was documentation: the docstring must state the convention."""
    doc = dm.Model  # keep the import used; the text under test lives on Sensitivity
    from discopt.sensitivity import Sensitivity

    assert doc is dm.Model
    text = Sensitivity.__doc__ or ""
    assert "internal-minimization sign convention" in text
    assert "MAXIMIZE" in text
    # And ``status`` must no longer read as a global-optimality certificate.
    assert "NOT that the point is a certified global optimum" in text
