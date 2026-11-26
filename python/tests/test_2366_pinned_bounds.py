"""#2366: a DAE fit whose states the dynamics hold exactly on ``lb=0`` stalls.

A state that starts at 0 and stays there until an input switches on has every
pre-input collocation value at 0 *and* determined by the collocation equations,
so POUNCE's interior-point method has no strictly feasible region there. The
model below is the smallest version of the issue's pulsed reactor: two coupled
states, ``a`` driven by an input that switches on at ``t = 1``, ``c`` fed by
``a``. Measured (pounce-solver 0.19.0, discopt defaults):

* ``lb=0``: ``local_limit`` (0.42 s), 24 columns per state pinned at 0, in
  elements 0-6 (the input switches on in element 6);
* ``lb=-1e-8``: ``local_optimal`` (0.03 s), the same objective to 1e-9.

The fix reports the signature: the unconverged solve warns, naming the variables
and finite elements, and attaches ``result.pinned_bounds``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
from discopt._pinned_bounds import (  # noqa: E402
    PinnedBoundReport,
    _ranges,
    equality_columns,
    pinned_bound_report,
)
from discopt.dae import ContinuousSet, DAEBuilder  # noqa: E402

NFE = 12  # elements of width 1/6 on [0, 2]; the input switches on at t = 1
SWITCH_ELEMENT = 6


def _pulse_fit(lb: float) -> dm.Model:
    m = dm.Model("pulse")
    k = m.continuous("k", lb=0.1, ub=10.0)
    cs = ContinuousSet("t", bounds=(0.0, 2.0), nfe=NFE, ncp=3, scheme="radau")
    b = DAEBuilder(m, cs)
    b.add_state("a", bounds=(lb, 2.0), initial=0.0)
    b.add_state("c", bounds=(lb, 2.0), initial=0.0)

    def rhs(t, s, alg, ctl):
        u = np.where(np.asarray(t.value) > 1.0, 1.0, 0.0)
        return {"a": u - k * s["a"] * (1 + s["c"]), "c": k * s["a"] - 0.5 * s["c"]}

    b.set_ode(rhs)
    b.discretize()
    a, c = b.get_state("a"), b.get_state("c")
    m.minimize((a[-1, -1] - 0.3) ** 2 + (c[-1, -1] - 0.2) ** 2)
    return m


def _solve(model, **opts):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = model.solve(solver="pounce", pounce_options={"print_level": 0, **opts})
    return r, [str(w.message) for w in caught if "#2366" in str(w.message)]


@pytest.mark.smoke
def test_lb_zero_stall_names_the_pinned_states_and_elements():
    r, msgs = _solve(_pulse_fit(0.0))
    assert r.status != "local_optimal", "the #2366 stall no longer reproduces; re-measure"
    report = r.pinned_bounds
    assert isinstance(report, PinnedBoundReport)
    assert len(msgs) == 1
    by_var = {g.variable: g for g in report.groups}
    assert set(by_var) == {"t_a", "t_c"}
    for g in by_var.values():
        assert g.side == "lower" and g.bounds == [0.0]
        # Pinned before the input switches on, and nowhere after the element it
        # switches on in: the report localises the degeneracy in time.
        assert 0 in g.leading_indices
        assert max(g.leading_indices) <= SWITCH_ELEMENT
        assert f"{g.variable}: {g.count} at lb=0" in msgs[0]
    assert report.n_columns == sum(g.count for g in report.groups)
    assert report.effective_relaxation == 0.0
    assert "lb=-1e-8" in msgs[0] and "result.pinned_bounds" in msgs[0]


@pytest.mark.smoke
def test_slightly_negative_bound_converges_without_a_report():
    """The documented remedy: same fit, ``lb=-1e-8``, converges and says nothing."""
    r, msgs = _solve(_pulse_fit(-1e-8))
    assert r.status == "local_optimal"
    assert r.pinned_bounds is None and not msgs
    ref, _ = _solve(_pulse_fit(0.0))
    # Same fit: the stalled run's objective agrees, it just never certified.
    assert ref.objective is not None
    assert r.objective == pytest.approx(ref.objective, abs=1e-6)


@pytest.mark.smoke
def test_relaxation_cap_at_constr_viol_tol_is_named():
    """``bound_relax_factor=1e-8`` under ``constr_viol_tol=1e-10`` relaxes by 1e-10
    (POUNCE/Ipopt cap it), which is why the issue's case D still stalled."""
    r, msgs = _solve(
        _pulse_fit(0.0), bound_relax_factor=1e-8, constr_viol_tol=1e-10, tol=1e-14, max_iter=60
    )
    assert r.status != "local_optimal"
    assert r.pinned_bounds is not None
    assert r.pinned_bounds.effective_relaxation == pytest.approx(1e-10)
    assert "capped at constr_viol_tol=1e-10" in msgs[0]


@pytest.mark.smoke
def test_unconverged_solve_off_the_bound_is_not_reported():
    """A limit stop whose iterate is nowhere near a bound gets no #2366 hint."""
    r, msgs = _solve(_pulse_fit(-0.05), tol=1e-14, max_iter=60)
    assert r.status != "local_optimal"
    assert r.pinned_bounds is None and not msgs


class _Structure:
    """Minimal evaluator stand-in: 3 rows over 4 columns."""

    n_variables = 4
    n_constraints = 3

    def jacobian_structure(self):
        # row 0 (equality): cols 0, 1; row 1 (inequality): col 2; row 2 (eq): col 3
        return np.array([0, 0, 1, 2]), np.array([0, 1, 2, 3])


@pytest.mark.smoke
def test_equality_columns_ignores_inequality_rows():
    cl = np.array([0.0, -np.inf, 1.0])
    cu = np.array([0.0, 0.0, 1.0])
    mask = equality_columns(_Structure(), cl, cu)
    assert mask.tolist() == [True, True, False, True]


@pytest.mark.smoke
def test_report_filters_fixed_inequality_only_and_interior_columns():
    class _V:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape
            self.size = int(np.prod(shape)) if shape else 1

    class _M:
        _variables = [_V("x", (2, 2)), _V("s", ())]

    #          x[0,0] x[0,1] x[1,0] x[1,1]  s    aux
    x = np.array([0.0, 1e-9, 0.5, 3.0 - 1e-7, 2.0, 1e-8])
    lb = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
    ub = np.array([0.0, 1.0, 1.0, 3.0, 5.0, 1.0])
    in_eq = np.array([True, True, True, True, False, True])
    rep = pinned_bound_report(_M(), x, lb, ub, in_eq)
    assert rep is not None
    got = {(g.variable, g.side): (g.count, g.leading_indices) for g in rep.groups}
    # x[0,0] is fixed (lb == ub), x[1,0] is interior, s is in no equality row.
    assert got == {
        ("x", "lower"): (1, [0]),
        ("x", "upper"): (1, [1]),
        ("<auxiliary>", "lower"): (1, [0]),
    }
    assert rep.n_columns == 3
    assert pinned_bound_report(_M(), x, lb, ub, np.zeros(6, dtype=bool)) is None
    with pytest.raises(ValueError, match="sizes differ"):
        pinned_bound_report(_M(), x[:-1], lb, ub, in_eq)


@pytest.mark.smoke
def test_ranges_compresses_runs():
    assert _ranges([0, 1, 2, 5, 7, 8]) == "0-2, 5, 7-8"
    assert _ranges([4]) == "4"
