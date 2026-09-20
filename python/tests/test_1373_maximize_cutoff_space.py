"""#1373: an incumbent cutoff must reach each kernel in the space that kernel reads.

The B&B tree, the LP relaxation rows and every evaluator carry the objective in
the internal *minimization* space -- ``-f`` for a MAXIMIZE model -- so
``tree.incumbent()[1]`` is ``-f(x_inc)``.  Two families of kernel consume a
cutoff, and they read it in **opposite** spaces:

* the Rust ``fbbt_with_cutoff`` / ``in_tree_presolve`` pair builds the cutoff row
  against the repr's own objective under the repr's declared
  ``objective_sense``, so it needs ``f(x_inc)`` -- the model's space;
* ``run_obbt_on_relaxation`` / ``dbbt_on_relaxation`` build it against
  ``relaxation._c``, which is already negated for a maximize, so they need the
  internal value.

Every solver call site passed the internal value to both.  For the Rust pair that
made the row ``f >= -f(x_inc)``, which on a maximize model with a *negative*
optimum is stricter than valid: it empties every child box, the tree closes at
the suboptimal incumbent, and the published dual bound sits BELOW the true
optimum -- a false ``optimal`` with ``gap_certified=True``.

The convention is now one space (internal) for every ``incumbent_cutoff``
parameter, converted at the Rust boundary by
:func:`discopt.modeling.core.repr_space_cutoff`, which also *verifies* the
converted value against the incumbent point and declines (no cutoff, a looser and
sound box) when the two spaces cannot be shown to agree.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.obbt import bootstrap_finite_bounds, run_obbt
from discopt._rust import model_to_repr
from discopt.modeling.core import objective_sense_sign, repr_space_cutoff

# ── The reported instance ────────────────────────────────────────────────────
# x1 integer, x0 = x1 +/- 84 inside [-60, 96], min(x1, x0) <= -17.3: 36 feasible
# points, of which x1 = -18, x0 = 66 maximizes -x1^2 at -324.
TRUE_OPT = -324.0
FALSE_INCUMBENT = -576.0  # what the diving heuristic injects first (x1 = +-24)


def _reported_model():
    m = dm.Model("i1373")
    m.continuous("x0", lb=-60.0, ub=96.0)
    m.integer("x1", lb=-48, ub=28)
    x0, x1 = m._variables
    m.subject_to(dm.abs(x1 - x0) == 84.0, name="c0")
    m.subject_to(dm.minimum(x1, x0) <= -17.3, name="c1")
    m.maximize(-(x1**2))
    return m


# Marked ``smoke`` as well as ``correctness``: the default ``addopts`` deselect
# ``correctness``, and this is the false-certificate class itself -- it has to run
# under the ``pytest -m smoke`` PR gate, where the command-line ``-m`` replaces
# that filter. ~1.4 s, marginally over the ``smoke`` budget, which the class earns.
@pytest.mark.smoke
@pytest.mark.correctness
@pytest.mark.pr_correctness
def test_maximize_certificate_does_not_fathom_the_optimum():
    """The whole certificate, not just the objective (the issue's own ask)."""
    r = _reported_model().solve(time_limit=60)

    assert r.objective == pytest.approx(TRUE_OPT, abs=1e-6), (
        f"returned {r.objective}; the in-tree presolve cutoff fathomed the optimum "
        f"and closed at the suboptimal incumbent {FALSE_INCUMBENT}"
    )
    # For a maximize model the dual bound is an UPPER bound: it must never fall
    # below the true optimum. `bound == -576` was the false certificate.
    assert r.bound is not None and np.isfinite(r.bound)
    assert r.bound >= TRUE_OPT - 1e-6, (
        f"dual bound {r.bound} is BELOW the true optimum {TRUE_OPT}: the "
        "certificate cuts the optimum out"
    )
    assert r.bound >= r.objective - 1e-6, "bound below incumbent for a maximize"


@pytest.mark.correctness
def test_in_tree_presolve_stride_zero_agrees():
    """The flag A/B from the issue: the reduction must be bound-neutral here."""
    on = _reported_model().solve(time_limit=60)
    off = _reported_model().solve(time_limit=60, in_tree_presolve_stride=0)
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    assert on.bound == pytest.approx(off.bound, rel=1e-9, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.correctness
def test_minimize_twin_unchanged():
    """Control: the same math under MINIMIZE, where internal == model space.

    A fix that negated unconditionally would break this.
    """
    m = dm.Model("i1373_min")
    m.continuous("x0", lb=-60.0, ub=96.0)
    m.integer("x1", lb=-48, ub=28)
    x0, x1 = m._variables
    m.subject_to(dm.abs(x1 - x0) == 84.0)
    m.subject_to(dm.minimum(x1, x0) <= -17.3)
    m.minimize(x1**2)
    r = m.solve(time_limit=60)
    assert r.objective == pytest.approx(-TRUE_OPT, abs=1e-6)
    assert r.bound is not None and r.bound <= r.objective + 1e-6


# ── The boundary itself ──────────────────────────────────────────────────────


@pytest.mark.smoke
def test_repr_kernel_cutoff_keeps_the_optimum_in_the_box():
    """Directly at the kernel: the internal value empties the box, the model
    value tightens it and keeps the optimum."""
    m = _reported_model()
    rp = model_to_repr(m, getattr(m, "_builder", None))
    lb = np.array([-60.0, -48.0])
    ub = np.array([96.0, 28.0])
    internal = -FALSE_INCUMBENT  # tree.incumbent()[1] == +576

    empty = rp.in_tree_presolve(lb, ub, node_depth=0, depth_stride=1, incumbent=internal)
    assert empty["ran"] and empty["infeasible"], (
        "the internal-space cutoff is expected to empty the box -- if it no "
        "longer does, this test no longer guards the reported failure"
    )

    converted = repr_space_cutoff(rp, internal)
    assert converted == pytest.approx(FALSE_INCUMBENT)
    good = rp.in_tree_presolve(lb, ub, node_depth=0, depth_stride=1, incumbent=converted)
    assert good["ran"] and not good["infeasible"]
    assert good["lb"][1] <= -18.0 <= good["ub"][1]
    assert good["lb"][0] <= 66.0 <= good["ub"][0]


@pytest.mark.smoke
def test_repr_space_cutoff_conversion_and_refusal():
    m = _reported_model()
    rp = model_to_repr(m, getattr(m, "_builder", None))
    assert rp.objective_sense == "maximize"

    # Converts, and verifies against the incumbent point (x1 = 24 -> f = -576).
    point = np.array([-60.0, 24.0])
    assert repr_space_cutoff(rp, 576.0, incumbent_point=point) == pytest.approx(-576.0)

    # Refuses when the point disagrees with the converted value: the spaces
    # cannot be shown to agree, so the reduction runs cutoff-free (sound).
    assert repr_space_cutoff(rp, 576.0, incumbent_point=np.array([66.0, -18.0])) is None

    # Non-finite / absent cutoffs decline rather than propagate.
    assert repr_space_cutoff(rp, None) is None
    assert repr_space_cutoff(rp, float("inf")) is None

    # A minimize repr is the identity, point-verified.
    mn = dm.Model("i1373_id")
    mn.continuous("x", lb=0.0, ub=4.0)
    (x,) = mn._variables
    mn.minimize(x * x)
    rn = model_to_repr(mn, getattr(mn, "_builder", None))
    assert rn.objective_sense == "minimize"
    assert repr_space_cutoff(rn, 9.0, incumbent_point=np.array([3.0])) == pytest.approx(9.0)


@pytest.mark.smoke
def test_conversion_is_the_one_sign_definition():
    """The conversion must be ``objective_sense_sign``, not a hand-written sign.

    #1299 is what happens when each site remembers on its own; this pins the
    repr-boundary conversion to that one definition for both senses.
    """
    mx = _reported_model()
    rx = model_to_repr(mx, getattr(mx, "_builder", None))
    assert repr_space_cutoff(rx, 576.0) == pytest.approx(objective_sense_sign(mx) * 576.0)

    mn = dm.Model("i1373_min_sign")
    mn.continuous("x", lb=0.0, ub=1.0)
    (x,) = mn._variables
    mn.minimize(x)
    rn = model_to_repr(mn, getattr(mn, "_builder", None))
    assert repr_space_cutoff(rn, 576.0) == pytest.approx(objective_sense_sign(mn) * 576.0)


# ── The OBBT-family row, which reads the OTHER space ────────────────────────
# max -2x - 3y  s.t.  x + y >= 5,  x, y in [0, 10]  ->  optimum -10 at (5, 0).
# An incumbent at (0, 5) has f = -15, i.e. internal +15. The cutoff row must be
# `2x + 3y <= 15` (x <= 7.5, y <= 5) -- active, and it keeps (5, 0). Negating the
# right-hand side for the maximize arm, as the code did while every caller handed
# it an internal value, gives `2x + 3y <= -15`: empty over the feasible region, so
# the cutoff was silently DEAD on every maximize model rather than helpful.
_LIN_OPT = np.array([5.0, 0.0])
_LIN_INTERNAL_INC = 15.0


def _linear_max(open_box=False):
    m = dm.Model("i1373_lin")
    m.continuous("x", lb=0.0, ub=np.inf if open_box else 10.0)
    m.continuous("y", lb=0.0, ub=np.inf if open_box else 10.0)
    x, y = m._variables
    m.subject_to(x + y >= 5.0)
    m.subject_to(x <= 10.0)
    m.subject_to(y <= 10.0)
    m.maximize(-2.0 * x - 3.0 * y)
    return m


def _keeps(lb, ub, point=_LIN_OPT):
    return bool(np.all(lb <= point + 1e-7) and np.all(ub >= point - 1e-7))


@pytest.mark.smoke
def test_run_obbt_internal_cutoff_is_active_and_keeps_the_optimum():
    r = run_obbt(
        _linear_max(),
        lb=np.array([0.0, 0.0]),
        ub=np.array([10.0, 10.0]),
        incumbent_cutoff=_LIN_INTERNAL_INC,
    )
    assert _keeps(r.tightened_lb, r.tightened_ub)
    # The row is live: 2x + 3y <= 15 pins x <= 7.5 and y <= 5.
    assert r.tightened_ub[0] == pytest.approx(7.5, abs=1e-5)
    assert r.tightened_ub[1] == pytest.approx(5.0, abs=1e-5)


@pytest.mark.smoke
def test_bootstrap_finite_bounds_internal_cutoff_finitizes():
    lb, ub, n_fin, _ = bootstrap_finite_bounds(
        _linear_max(open_box=True),
        np.array([0.0, 0.0]),
        np.array([np.inf, np.inf]),
        incumbent_cutoff=_LIN_INTERNAL_INC,
    )
    assert n_fin == 2, "the internal-space cutoff must finitize both open bounds"
    assert _keeps(lb, ub)
    assert np.all(np.isfinite(ub))


@pytest.mark.smoke
def test_old_maximize_rhs_was_dead_not_merely_different():
    """Falsification arm: the value the old maximize branch computed.

    The old code built ``rhs = -incumbent_cutoff``; with an internal ``+15`` that
    is ``-15``, which is what passing ``-15`` to the fixed row reproduces
    exactly. It yields no tightening at all -- the polytope is empty, so every
    OBBT LP is infeasible and the pass returns the input box.
    """
    r = run_obbt(
        _linear_max(),
        lb=np.array([0.0, 0.0]),
        ub=np.array([10.0, 10.0]),
        incumbent_cutoff=-_LIN_INTERNAL_INC,
    )
    assert _keeps(r.tightened_lb, r.tightened_ub)
    assert r.tightened_ub[0] == pytest.approx(10.0, abs=1e-9)
    assert r.tightened_ub[1] == pytest.approx(10.0, abs=1e-9)
