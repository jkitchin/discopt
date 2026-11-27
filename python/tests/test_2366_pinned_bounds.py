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

The first fix reported the signature: the unconverged solve warns, naming the
variables and finite elements, and attaches ``result.pinned_bounds``. The second
acts on it: the route re-solves once with the pinned variables' bounds moved
outward by a 1e-8 slab, projects the point back onto the declared box, and adopts
it only when it converged and the projection passes discopt's 1e-6 feasibility
check. ``DISCOPT_POUNCE_PINNED_RETRY=0`` restores the report-only behaviour, which
the stall tests below pin.
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


@pytest.fixture
def no_retry(monkeypatch):
    monkeypatch.setenv("DISCOPT_POUNCE_PINNED_RETRY", "0")


@pytest.mark.smoke
def test_lb_zero_stall_names_the_pinned_states_and_elements(no_retry):
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
    assert r.algorithm_route == "pounce:nlp"


@pytest.mark.smoke
def test_lb_zero_stall_is_retried_on_a_slab_and_converges():
    """Default: the stalled ``lb=0`` fit is re-solved on a 1e-8 slab and converges to
    the ``lb=-1e-8`` objective, with the point inside the box the user declared."""
    m = _pulse_fit(0.0)
    r, msgs = _solve(m)
    assert r.status == "local_optimal"
    assert len(msgs) == 1 and "re-solved once" in msgs[0]
    assert r.algorithm_route == "pounce:nlp [#2366 bound-slab retry: t_a lb, t_c lb]"
    # The first solve's report is kept: it says why the retry ran.
    assert isinstance(r.pinned_bounds, PinnedBoundReport)
    assert {g.variable for g in r.pinned_bounds.groups} == {"t_a", "t_c"}
    # Projected: nothing below the declared lb=0, not even by the slab's 1e-8.
    for name in ("t_a", "t_c"):
        assert float(np.min(r.x[name])) >= 0.0
    assert r.bound is None and not r.gap_certified
    ref, ref_msgs = _solve(_pulse_fit(-1e-8))
    assert ref.status == "local_optimal" and not ref_msgs
    assert r.objective == pytest.approx(ref.objective, abs=1e-6)
    # The reported objective is the objective AT the reported (projected) point.
    a, c = m._variables[1], m._variables[2]
    assert r.objective == pytest.approx(
        (float(r.x[a.name][-1, -1]) - 0.3) ** 2 + (float(r.x[c.name][-1, -1]) - 0.2) ** 2,
        abs=1e-12,
    )


@pytest.mark.smoke
def test_slightly_negative_bound_converges_without_a_report():
    """The documented remedy: same fit, ``lb=-1e-8``, converges and says nothing."""
    r, msgs = _solve(_pulse_fit(-1e-8))
    assert r.status == "local_optimal"
    assert r.pinned_bounds is None and not msgs
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DISCOPT_POUNCE_PINNED_RETRY", "0")
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
    # The retry ran under the same 60-iteration cap and tol=1e-14, did not
    # converge, and was not adopted: the first solve's result is reported.
    assert "did not help" in msgs[0]
    assert r.algorithm_route == "pounce:nlp"


@pytest.mark.smoke
def test_unconverged_solve_off_the_bound_is_not_reported():
    """A limit stop whose iterate is nowhere near a bound gets no #2366 hint."""
    r, msgs = _solve(_pulse_fit(-0.05), tol=1e-14, max_iter=60)
    assert r.status != "local_optimal"
    assert r.pinned_bounds is None and not msgs


@pytest.mark.smoke
def test_skip_convex_check_local_nlp_reports_too():
    """The ``skip_convex_check`` local-NLP arm (where ``nlp_solver="ipopt"`` runs a
    single solve) attaches the same report. Run on its POUNCE backend, which shares
    the code path; at defaults its bound tightening lets ``lb=0`` converge, so an
    iteration cap forces the stall (measured: 48 pinned columns at ``max_iter=15``,
    none for ``lb=-0.05``)."""

    def run(lb):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = _pulse_fit(lb).solve(
                skip_convex_check=True,
                time_limit=60,
                pounce_options={"max_iter": 15, "print_level": 0},
            )
        return r, [str(w.message) for w in caught if "#2366" in str(w.message)]

    r, msgs = run(0.0)
    assert r.algorithm_route.startswith("local-nlp:")
    assert r.status != "feasible"
    assert isinstance(r.pinned_bounds, PinnedBoundReport)
    assert {g.variable for g in r.pinned_bounds.groups} <= {"t_a", "t_c"}
    assert len(msgs) == 1 and msgs[0].startswith("The local pounce NLP stopped")
    assert "pounce_options={'bound_relax_factor': 1e-8" in msgs[0]
    off, off_msgs = run(-0.05)
    assert off.pinned_bounds is None and not off_msgs


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
