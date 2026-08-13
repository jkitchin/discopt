"""Tests for :mod:`discopt.solvers._dfo_common`, shared by ``direct`` and ``surrogate``.

Two properties are checked here, and neither had a test before #1010 extracted
the module:

* **The two backends see identical values.** Both docstrings asserted it, nothing
  measured it. The extraction makes it true by construction, so this test is the
  guard against a future caller re-forking the oracle — it drives the *backends'*
  search objects, not ``build_oracle`` twice, so it fails if either backend stops
  routing through the shared implementation.
* **The merit rule, directly.** ``test_direct_units`` covered phases A/B and the
  ``ε_cons`` band; ``test_surrogate_units`` covered phases A/B and the non-finite
  fill. Each was a partial covering of one copy. The rule now has one home and
  one set of tests, with ``finite_fill`` exercised in both settings.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt import modeling as dm
from discopt.solvers._dfo_common import glce_merit, glce_merit_scalar

pytestmark = pytest.mark.unit


# ── the merit rule (DIRECT-GLce) ─────────────────────────────────────────────


def _merit(f, v, best, eps=1e-6, *, finite_fill=False):
    return glce_merit(
        np.asarray(f, float), np.asarray(v, float), best, eps, finite_fill=finite_fill
    )


def test_phase_a_ranks_by_violation_before_any_feasible_point():
    """With nothing feasible yet, the search minimizes total violation.

    The objective must not enter at all: the second point has a far better ``f``
    and must still lose on violation alone.
    """
    merit = _merit(f=[100.0, -100.0], v=[5.0, 9.0], best=None)
    assert merit[0] < merit[1], merit
    np.testing.assert_allclose(merit, [5.0, 9.0])


def test_phase_b_denies_credit_to_an_infeasible_low_objective():
    """An infeasible point cannot outrank the incumbent by having a lower objective.

    The ``|f - f_min|`` term is what removes that credit, and it needs no penalty
    weight to be tuned.
    """
    merit = _merit(f=[10.0, -1e6], v=[0.0, 3.0], best=10.0)
    assert merit[0] < merit[1], merit
    # Feasible point ranks by its objective; the infeasible one by the full rule.
    assert merit[0] == pytest.approx(10.0)
    assert merit[1] == pytest.approx(-1e6 + 3.0 + abs(-1e6 - 10.0))


def test_the_eps_cons_band_is_treated_as_feasible():
    """The 'ce' refinement: no penalty discontinuity right at the feasible boundary.

    Inside the band the merit is exactly ``f``; a hair outside it the penalty
    switches on. Both arms are asserted so the test fails if the band widens to
    everything or collapses to nothing.
    """
    inside = _merit(f=[4.0], v=[1e-6], best=5.0, eps=1e-3)
    assert inside[0] == pytest.approx(4.0)

    outside = _merit(f=[4.0], v=[1e-2], best=5.0, eps=1e-3)
    assert outside[0] == pytest.approx(4.0 + 1e-2 + 1.0)


def test_the_band_boundary_is_inclusive():
    """``v == eps_cons`` counts as feasible, matching ``_offer`` in both backends.

    The incumbent bookkeeping accepts ``viol <= eps_cons``; if the merit used a
    strict ``<`` the point that became the incumbent would still be ranked as
    infeasible.
    """
    assert _merit(f=[4.0], v=[1e-3], best=5.0, eps=1e-3)[0] == pytest.approx(4.0)


def test_an_unconstrained_model_ranks_by_the_objective_alone():
    """Zero violation everywhere reduces the rule to ``f``."""
    merit = _merit(f=[3.0, 1.0, 2.0], v=[0.0, 0.0, 0.0], best=1.0)
    np.testing.assert_allclose(merit, [3.0, 1.0, 2.0])


# -- the finite_fill choice ---------------------------------------------------


def test_finite_fill_false_leaves_an_undefined_objective_at_infinity():
    """DIRECT's setting: the merit is only compared, so ``+inf`` loses everything.

    This is the honest ranking for "the black box had no value here", and it
    costs nothing because the value is never fitted.
    """
    merit = _merit(f=[1.0, 3.0, np.inf], v=[0.0, 0.0, 0.0], best=1.0, finite_fill=False)
    assert np.isinf(merit[2])
    assert np.isfinite(merit[:2]).all()


def test_finite_fill_true_maps_an_undefined_objective_to_a_finite_worst_case():
    """The surrogate's setting: an infinite right-hand side has no fit.

    ``+inf`` makes every interpolation coefficient ``nan``; dropping the point
    throws away the one thing it does tell us, which is that the region is bad.
    Mapping it to the worst finite merit plus the observed spread keeps both.
    """
    merit = _merit(f=[1.0, 3.0, np.inf], v=[0.0, 0.0, 0.0], best=1.0, finite_fill=True)
    assert np.all(np.isfinite(merit))
    assert merit[2] > merit[1]
    assert merit[2] == pytest.approx(3.0 + (3.0 - 1.0) + 1.0)


def test_finite_fill_true_with_no_finite_merit_at_all_is_well_posed():
    """Every point undefined leaves no spread to borrow; the fit must still exist.

    Zeros are uninformative but finite, which is the only property the
    interpolation system needs. ``nan`` or ``inf`` here would propagate into every
    coefficient.
    """
    merit = _merit(f=[np.inf, np.inf], v=[0.0, 0.0], best=1.0, finite_fill=True)
    np.testing.assert_allclose(merit, [0.0, 0.0])


def test_finite_fill_does_not_disturb_an_all_finite_merit():
    """The fill is a repair, not a transform: with nothing to repair it is identity."""
    args = dict(f=[3.0, 1.0, 7.0], v=[0.0, 2.0, 0.0], best=1.0)
    np.testing.assert_allclose(_merit(**args, finite_fill=True), _merit(**args, finite_fill=False))


def test_an_empty_point_set_returns_an_empty_merit():
    """Called before the first evaluation lands; must not raise on either fill."""
    for fill in (True, False):
        assert _merit(f=[], v=[], best=None, finite_fill=fill).shape == (0,)
        assert _merit(f=[], v=[], best=1.0, finite_fill=fill).shape == (0,)


# -- the scalar wrapper -------------------------------------------------------


@pytest.mark.parametrize("best", [None, 10.0])
@pytest.mark.parametrize("fval,viol", [(4.0, 0.0), (4.0, 1e-8), (4.0, 3.0), (np.inf, 0.0)])
def test_the_scalar_wrapper_agrees_with_the_array_form(fval, viol, best):
    """``glce_merit_scalar`` must be the array rule at ``n=1``, not a second rule.

    That is the drift the shared module exists to prevent, so it is asserted
    rather than assumed — including on the ``+inf`` arm, where a hand-written
    scalar copy would be easy to get subtly different.
    """
    expected = _merit(f=[fval], v=[viol], best=best, finite_fill=False)[0]
    got = glce_merit_scalar(fval, viol, best, 1e-6)
    if np.isinf(expected):
        assert np.isinf(got) and np.sign(got) == np.sign(expected)
    else:
        assert got == pytest.approx(expected)


# ── the two backends see identical values ────────────────────────────────────


def _opaque_constrained_model() -> dm.Model:
    """A model whose objective and constraint bodies are both opaque ``dm.custom``.

    Opaque on purpose: this is the regime both backends exist for, and it is the
    case where a divergence in the oracle would be least visible — there is no
    algebraic form to check the evaluated value against.
    """
    m = dm.Model("dfo_oracle_parity")
    x = m.continuous("x", shape=2, lb=-2.0, ub=3.0)
    k = m.integer("k", lb=0, ub=4)
    m.minimize(dm.custom(lambda v: v[0] ** 2 + 3.0 * v[1], name="obj")(x))
    # Two constraints with different bound shapes so cl and cu are both exercised.
    m.subject_to(dm.custom(lambda v: v[0] * v[1], name="c_bilinear")(x) <= 1.0)
    m.subject_to(dm.custom(lambda a, b: a + b, name="c_couple")(x[0], k) >= -1.0)
    return m


def _sample_points(rng: np.random.Generator, n_vars: int, n_points: int) -> list[np.ndarray]:
    """Points spanning feasible, infeasible and boundary regions of the box."""
    return [rng.uniform(-2.0, 3.0, size=n_vars) for _ in range(n_points)]


def test_the_two_backends_evaluate_a_point_identically():
    """The property the duplication was trusted to preserve and nothing checked.

    Routed through each backend's *own* entry point (``direct.build_oracle`` and
    ``surrogate.build_oracle`` as each module imports it), so it fails if either
    backend stops using the shared implementation — not just if the shared one
    changes. Values must match exactly, not approximately: they come from the same
    evaluator, so any difference at all is plumbing.
    """
    from discopt.solvers import direct as direct_mod
    from discopt.solvers import surrogate as surrogate_mod

    model = _opaque_constrained_model()
    d_oracle, d_n, d_mask = direct_mod.build_oracle(model, log_prefix="DIRECT")
    s_oracle, s_n, s_mask = surrogate_mod.build_oracle(model, log_prefix="surrogate")

    assert d_n == s_n
    np.testing.assert_array_equal(d_mask, s_mask)
    assert d_mask.any(), "the parity model must exercise the integer-mask path"

    rng = np.random.default_rng(20260813)
    compared = 0
    saw_feasible = False
    saw_infeasible = False
    for x in _sample_points(rng, d_n, 40):
        f_d, v_d = d_oracle(x)
        f_s, v_s = s_oracle(x)
        assert f_d == f_s, (x, f_d, f_s)
        assert v_d == v_s, (x, v_d, v_s)
        compared += 1
        saw_feasible |= v_d == 0.0
        saw_infeasible |= v_d > 0.0

    # CLAUDE.md §6: prove the probe fired, and that it covered both verdicts --
    # a run that only ever saw feasible points would not test the violation sum.
    assert compared == 40, compared
    assert saw_feasible and saw_infeasible, (saw_feasible, saw_infeasible)


def test_the_two_backends_agree_on_an_unconstrained_model():
    """The ``cl is None`` branch, which the constrained parity test never reaches."""
    from discopt.solvers import direct as direct_mod
    from discopt.solvers import surrogate as surrogate_mod

    m = dm.Model("dfo_oracle_parity_unconstrained")
    x = m.continuous("x", shape=2, lb=-1.0, ub=1.0)
    m.minimize(dm.custom(lambda v: v[0] ** 2 + v[1] ** 2, name="obj")(x))

    d_oracle, d_n, _ = direct_mod.build_oracle(m, log_prefix="DIRECT")
    s_oracle, _, _ = surrogate_mod.build_oracle(m, log_prefix="surrogate")

    rng = np.random.default_rng(7)
    compared = 0
    for x_pt in _sample_points(rng, d_n, 10):
        assert d_oracle(np.clip(x_pt, -1.0, 1.0)) == s_oracle(np.clip(x_pt, -1.0, 1.0))
        compared += 1
    assert compared == 10, compared


@pytest.mark.parametrize(
    "cl,cu",
    [
        ([-1.0], [1.0, 1.0]),  # cu too long
        ([-1.0, -1.0, -1.0], [1.0, 1.0]),  # cl too long
    ],
)
def test_a_constraint_bound_length_mismatch_is_refused(monkeypatch, cl, cu):
    """The guard that had to be applied twice — now once, so it cannot diverge again.

    A length mismatch against ``g`` is a *legal* numpy broadcast, so it produces a
    silently wrong violation — a wrong feasibility verdict, not a crash. Both
    directions are covered because a one-sided check would pass one of them.

    ``_infer_constraint_bounds`` is patched at its definition site rather than on
    ``_dfo_common``, since ``build_oracle`` imports it inside the function body.
    """
    import discopt.solver as solver_mod
    from discopt.solvers._dfo_common import build_oracle as shared_build_oracle

    monkeypatch.setattr(
        solver_mod,
        "_infer_constraint_bounds",
        lambda model, evaluator: (np.array(cl), np.array(cu)),
    )
    with pytest.raises(ValueError, match="constraint bounds do not match the evaluator"):
        shared_build_oracle(_opaque_constrained_model(), log_prefix="DIRECT")


def test_matched_constraint_bounds_are_not_refused():
    """No-false-refusal control for the guard above."""
    from discopt.solvers._dfo_common import build_oracle as shared_build_oracle

    oracle, n_vars, _ = shared_build_oracle(_opaque_constrained_model(), log_prefix="DIRECT")
    fval, viol = oracle(np.zeros(n_vars))
    assert np.isfinite(fval) and np.isfinite(viol)
