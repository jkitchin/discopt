"""Engage the LP relaxer for general transcendental nonlinearity.

Models whose only nonlinearity is a general transcendental term (sqrt, exp,
sin/cos, ...) were routed away from the McCormick LP relaxer, so spatial B&B got
*no* dual bound and could not prove optimality (minlptests_nlp_003: 179 nodes,
status "feasible", root bound None). ``build_milp_relaxation`` already emits a
valid polyhedral outer approximation for these terms, so engaging the relaxer
yields a sound dual bound. These tests pin that the relaxer now engages and the
resulting bound is valid (never beyond the true optimum).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds


def _root_bound(model):
    rx = MccormickLPRelaxer(model)
    lb, ub = flat_variable_bounds(model)
    return rx.solve_at_node(lb, ub).lower_bound


def test_relaxer_engages_on_general_transcendental():
    m = dm.Model("sqrt_only")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.minimize(-dm.sqrt(x + 0.1))
    assert MccormickLPRelaxer(m).has_relaxable_nonlinearity is True


@pytest.mark.slow
def test_transcendental_model_proves_optimality_with_valid_bound():
    # nlp_003-style: sqrt objective, exp + sin^2 constraints.
    m = dm.Model("nlp003")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(dm.sqrt(x + 0.1))
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    res = m.solve(time_limit=60)
    assert res.status == "optimal"  # was "feasible" (no bound) before
    # The dual bound is valid: for a maximization it never *under*-states the
    # optimum, and at proven optimality it matches the objective.
    assert res.bound is not None
    assert float(res.bound) >= float(res.objective) - 1e-4
    assert res.node_count <= 50  # was 179 with no bound to prune on


@pytest.mark.slow
def test_exp_objective_minimization_bound_is_valid():
    # min exp(x) over [-2, 2]: optimum exp(-2) ~ 0.1353; relaxer must not
    # over-state the bound (a valid lower bound is <= the optimum).
    m = dm.Model("exp_min")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(dm.exp(x))
    res = m.solve(time_limit=30)
    assert res.status == "optimal"
    assert abs(float(res.objective) - 0.13533528) < 1e-4
    if res.bound is not None:
        assert float(res.bound) <= float(res.objective) + 1e-4  # valid lower bound


# ── FBBT rescue of transcendental envelopes over declared-unbounded args (#219) ──
#
# When a univariate operator's argument has a non-finite declared box but is
# finitely bounded via implied constraints (FBBT), the relaxer used to *drop* the
# constraint instead of building an envelope. ``_collect_univariate_relaxations``
# now retries over an FBBT-tightened box before giving up.


def _log_unbounded_but_fbbt_bounded():
    # log of an affine arg whose variable is declared ub=+inf, but a row bounds it
    # finitely. Before the fix the log row was dropped (root bound trivial / loose);
    # after, FBBT derives x<=5 and the envelope activates.
    m = dm.Model("log_rescue")
    x = m.continuous("x", lb=0.0)  # ub = +inf
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.maximize(y)
    m.subject_to(y <= dm.log(1 + x))
    m.subject_to(x <= 5.0)
    return m


def test_fbbt_rescues_dropped_log_envelope_and_bound_is_sound():
    m = _log_unbounded_but_fbbt_bounded()
    bound = _root_bound(m)
    # The relaxer builds a *finite* root bound (constraint not dropped).
    assert bound is not None
    # Sound: for the maximize lifted to minimize(-y), the bound never over-states
    # the true optimum max y = log(6) = 1.7918, i.e. lower_bound <= -1.7918.
    assert float(bound) <= -1.7918 + 1e-6


def test_genuinely_unbounded_arg_is_not_falsely_bounded():
    # No row bounds x, so FBBT cannot derive a finite argument box; the rescue
    # must *not* fire and must not invent a bound (soundness over coverage).
    m = dm.Model("log_truly_unbounded")
    x = m.continuous("x", lb=0.0)  # ub = +inf, nothing implies a finite bound
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.maximize(y)
    m.subject_to(y <= dm.log(1 + x))
    bound = _root_bound(m)
    # The log row stays dropped; y is bounded only by its box, so the relaxation
    # cannot certify a finite log-implied bound (no false tightening).
    assert bound is None or float(bound) <= -1.0 + 1e-9


@pytest.mark.slow
def test_gkocis_certifies_after_fbbt_rescue():
    import glob

    fns = glob.glob("python/tests/data/**/gkocis.nl", recursive=True)
    if not fns:
        pytest.skip("gkocis.nl not available")
    m = dm.from_nl(fns[0])
    res = m.solve(time_limit=60, gap_tolerance=1e-4)
    # Correct optimum (MINLPLib -1.923) AND now provably certified.
    assert abs(float(res.objective) - (-1.923)) < 1e-2
    assert res.gap_certified is True
