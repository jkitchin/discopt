"""#2143 (from #1917): heuristic sub-NLPs under hull fail on fixings that have points.

Under ``gdp_method="hull"`` a sub-NLP that fixes an indicator to 0 leaves that
disjunct's disaggregated copies on their whole box, held at 0 only by the linking
rows ``lb*y <= v <= ub*y``. The interior-point solve then approaches ``v = 0``
through eps-perspective rows evaluated at ``yhat ~ eps``. Measured on GDPlib
``syngas`` at bdc54a48, first root sub-NLP with its integers set to the big-M
incumbent's (known-feasible) configuration: infeasible at ITERATION_LIMIT after
300 iterations (4.5 s), and still after 3000 (41 s).

``DISCOPT_SUBNLP_FBBT_BOX`` (re-solve a failed sub-NLP on its node-FBBT box, with a
``Restoration_Failed`` constant-row retry) was RETIRED in the PR that introduced it,
because its 120 s panel was not cert-clean; see the row in
``docs/dev/flag-retirement-audit.md``. #2143's ``DISCOPT_SUBNLP_REDUCED`` (default
OFF) instead poses the ONE sub-NLP on the reduced problem: columns node FBBT pins
are fixed, rows over fixed columns only are checked at the fixed point and dropped.
The legacy defect stays pinned under ``=0``; the fix is asserted under ``=1``; the
retired retry's absence is pinned so node NLPs stay unchanged.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solvers.nlp_pounce as nlp_pounce  # noqa: E402
from discopt._relax import primal_heuristics as PH  # noqa: E402
from discopt._relax.gdp_reformulate import reformulate_gdp  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator  # noqa: E402
from discopt.solver import (  # noqa: E402
    _BoundOverrideEvaluator,
    _extract_variable_info,
    _make_evaluator,
)
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402

_RESTORATION_FAILED = -2


def _chain(n: int = 3):
    """``n`` two-term disjunctions, one with ``exp``, the other with ``log``."""
    m = dm.Model("hull_chain")
    xs = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(n)]
    zs = [m.continuous(f"z{i}", lb=-10, ub=10) for i in range(n)]
    for i in range(n):
        m.either_or(
            [
                [dm.exp(xs[i]) <= 3 + zs[i], xs[i] * xs[i] + zs[i] * zs[i] <= 20],
                [xs[i] >= 6, dm.log(1 + xs[i]) + zs[i] >= 2.5],
            ],
            name=f"d{i}",
        )
    m.minimize(sum((xs[i] - 5) ** 2 + zs[i] for i in range(n)))
    return m


def _configurations(low, n):
    """One start per one-hot configuration, continuous columns at their box midpoint."""
    lb, ub = PH._get_variable_bounds(low)
    names = []
    for v in low._variables:
        names += [v.name] * v.size
    ind = [i for i, nm in enumerate(names) if PH._get_integer_mask(low)[i]]
    assert len(ind) == 2 * n, names
    mid = np.where(np.isfinite(lb) & np.isfinite(ub), 0.5 * (lb + ub), 0.0)
    for pick in itertools.product((0, 1), repeat=n):
        x = mid.copy()
        for k in range(n):
            x[ind[2 * k]] = 1.0 - pick[k]
            x[ind[2 * k + 1]] = float(pick[k])
        yield x


def _feasible_count(n: int = 3) -> tuple[int, int]:
    low = reformulate_gdp(_chain(n), "hull")
    found = tried = 0
    for x in _configurations(low, n):
        tried += 1
        out = PH.subnlp(low, x)
        if out is not None:
            found += 1
            # Whatever subnlp returns must pass the incumbent verifier on the FULL
            # lowered model (every row, bounds, integrality), independently of
            # subnlp's own check.
            assert np.all(np.isfinite(out[0]))
            assert verify_point(low, out[0]).ok
    return found, tried


def test_every_hull_configuration_subnlp_finds_its_point(monkeypatch):
    # #2143: on the reduced pose every one of the 8 configurations (each has a
    # feasible point) returns a verified point. Measured 8/8 ON vs 1/8 OFF.
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    found, tried = _feasible_count()
    assert tried == 8
    assert found == tried


def test_hull_subnlp_fails_on_some_configurations(monkeypatch):
    # Pins the measured legacy defect (flag OFF): on the original pose POUNCE stops
    # at Invalid_Number_Detected (``exp``/``log`` at yhat ~ eps) on these fixings.
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "0")
    found, tried = _feasible_count()
    assert tried == 8
    assert found < tried


def _one_constant_row():
    """``x`` fixed at 1 by its box with one row on ``x`` alone; ``y`` free."""
    m = dm.Model("const_row")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.minimize((y - 2.0) * (y - 2.0) + x)
    m.subject_to(x == 1.0)
    m.subject_to(x + y <= 4.0)
    ev = _make_evaluator(m)
    _, lb, ub, _, _ = _extract_variable_info(m)
    lb, ub = lb.copy(), ub.copy()
    lb[0] = ub[0] = 1.0
    return _BoundOverrideEvaluator(ev, lb, ub), [(0.0, 0.0), (-np.inf, 0.0)]


def test_restoration_failure_is_not_retried(monkeypatch):
    # The retired flag's constant-row retry on Restoration_Failed is gone: a node
    # NLP that fails restoration is reported as failed, unchanged from before #1917.
    real = nlp_pounce._run_solve
    calls = []

    def fake(problem, x0, warm_start, solve_report=False):
        calls.append(1)
        if len(calls) == 1:
            return x0.copy(), {"status": _RESTORATION_FAILED, "obj_val": np.nan}, None
        return real(problem, x0, warm_start, solve_report)

    monkeypatch.setattr(nlp_pounce, "_run_solve", fake)
    ev, bounds = _one_constant_row()
    r = nlp_pounce.solve_nlp(ev, np.array([1.0, 1.0]), constraint_bounds=bounds)
    assert len(calls) == 1
    assert r.raw_status == _RESTORATION_FAILED and r.status == SolveStatus.ERROR


# ---------------------------------------------------------------- #2143 reduced pose


class _Recorder:
    """A backend that records what it was handed and solves nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, evaluator, x0, **kwargs):
        lb, ub = evaluator.variable_bounds
        self.calls.append(
            {
                "evaluator": evaluator,
                "n_constraints": int(evaluator.n_constraints),
                "lb": np.array(lb, dtype=np.float64),
                "ub": np.array(ub, dtype=np.float64),
                "x0": np.array(x0, dtype=np.float64),
                "kwargs": dict(kwargs),
            }
        )
        raise RuntimeError("recorder: no NLP result")


def _hull_toy():
    low = reformulate_gdp(_chain(3), "hull")
    return low, make_evaluator(low), next(iter(_configurations(low, 3)))


def test_the_flag_is_default_off(monkeypatch):
    monkeypatch.delenv("DISCOPT_SUBNLP_REDUCED", raising=False)
    assert PH._subnlp_reduced_enabled() is False
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    assert PH._subnlp_reduced_enabled() is True


def test_the_reduced_pose_pins_columns_and_drops_constant_rows(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    low, ev, x = _hull_toy()
    lb0, ub0 = (np.array(a, dtype=np.float64) for a in ev.variable_bounds)
    rec = _Recorder()
    assert PH.subnlp(low, x, backend=rec, evaluator=ev) is None
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["evaluator"] is not ev
    assert isinstance(call["evaluator"], PH._RowSubsetEvaluator)
    assert 0 < call["n_constraints"] < ev.n_constraints
    bounds = call["kwargs"]["constraint_bounds"]
    assert len(bounds) == call["n_constraints"]
    assert len(call["evaluator"].evaluate_constraints(call["x0"])) == len(bounds)
    int_mask = PH._get_integer_mask(low)
    pinned = (call["lb"] == call["ub"]) & (lb0 != ub0) & ~int_mask
    # the off disjuncts' disaggregated copies, pinned to exactly 0
    assert pinned.sum() > 0
    np.testing.assert_array_equal(call["lb"][pinned], 0.0)
    np.testing.assert_array_equal(call["x0"][pinned], 0.0)
    # no column FBBT does not pin is narrowed
    free = ~pinned & ~int_mask
    np.testing.assert_array_equal(call["lb"][free], lb0[free])
    np.testing.assert_array_equal(call["ub"][free], ub0[free])


def test_the_flag_off_pose_is_the_legacy_call(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "0")
    low, ev, x = _hull_toy()
    rec = _Recorder()
    PH.subnlp(low, x, backend=rec, evaluator=ev)
    assert len(rec.calls) == 1
    assert rec.calls[0]["evaluator"] is ev
    assert set(rec.calls[0]["kwargs"]) == {"options"}


def _unpinned_model():
    """With ``y = 0`` the row leaves ``x`` its whole box: FBBT pins nothing."""
    m = dm.Model("no_pin")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.binary("y")
    m.minimize(x + y)
    m.subject_to(x >= 0.5 * y)
    return m


def test_a_fixing_with_no_pin_is_posed_exactly_as_before(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    m = _unpinned_model()
    ev = make_evaluator(m)
    rec = _Recorder()
    PH.subnlp(m, np.array([0.5, 0.0]), backend=rec, evaluator=ev)
    assert len(rec.calls) == 1
    assert rec.calls[0]["evaluator"] is ev
    # no constraint_bounds keyword: a backend with the legacy signature still works
    assert set(rec.calls[0]["kwargs"]) == {"options"}


def _violated_constant_row_model():
    """``b = 0`` pins ``x`` to 0 through a linear row, where ``cos(3x) <= 0.5`` fails.

    Node FBBT does not propagate the violation through ``cos``, so the fixing is
    not refuted before the reduced pose sees the row as a constant.
    """
    m = dm.Model("violated_constant_row")
    x = m.continuous("x", lb=0.0, ub=5.0)
    w = m.continuous("w", lb=0.0, ub=5.0)
    b = m.binary("b")
    m.minimize(x + w + b)
    m.subject_to(x <= 5.0 * b)
    m.subject_to(dm.cos(3.0 * x) <= 0.5)
    m.subject_to(w >= 1.0)
    return m


def test_a_violated_constant_row_refuses_without_a_solve(monkeypatch):
    m = _violated_constant_row_model()
    ev = make_evaluator(m)
    x = np.array([1.0, 1.0, 0.0])
    # Precondition: FBBT pins x and does NOT refute the fixing.
    with m.saved_bounds():
        m._variables[2].lb = np.array(0.0)
        m._variables[2].ub = np.array(0.0)
        _, _, t_lb, t_ub, infeasible = PH._implied_box(ev)
    assert not infeasible
    assert t_lb[0] == t_ub[0] == 0.0
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    rec = _Recorder()
    assert PH.subnlp(m, x, backend=rec, evaluator=ev) is None
    assert rec.calls == []
    # flag OFF: the legacy pose spends a solve on the same fixing
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "0")
    rec = _Recorder()
    assert PH.subnlp(m, x, backend=rec, evaluator=ev) is None
    assert len(rec.calls) == 1
    # the other fixing is feasible and still found on the reduced pose
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    got = PH.subnlp(m, np.array([1.0, 1.0, 1.0]), evaluator=ev)
    assert got is not None and verify_point(m, got[0]).ok


def test_the_model_bounds_are_restored_after_a_reduced_solve(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    low, ev, x = _hull_toy()
    before = [(np.array(v.lb).copy(), np.array(v.ub).copy()) for v in low._variables]
    assert PH.subnlp(low, x, evaluator=ev) is not None
    checked = 0
    for (lb0, ub0), v in zip(before, low._variables):
        np.testing.assert_array_equal(lb0, np.array(v.lb))
        np.testing.assert_array_equal(ub0, np.array(v.ub))
        checked += 1
    assert checked == len(low._variables) > 0


def test_the_row_subset_evaluator_matches_the_base_rows():
    low, ev, x = _hull_toy()
    m, n = ev.n_constraints, ev.n_variables
    keep = np.arange(0, m, 2)
    lb, ub = (np.array(a, dtype=np.float64) for a in ev.variable_bounds)
    sub = PH._RowSubsetEvaluator(ev, keep, lb, ub)
    assert sub.n_constraints == keep.size
    np.testing.assert_array_equal(sub.evaluate_constraints(x), ev.evaluate_constraints(x)[keep])
    # Jacobian values scattered through the subset's structure equal the base's rows.
    rows, cols = sub.jacobian_structure()
    dense = np.zeros((keep.size, n))
    np.add.at(dense, (rows, cols), sub.evaluate_jacobian_values(x))
    np.testing.assert_allclose(dense, np.asarray(ev.evaluate_jacobian(x))[keep], rtol=0, atol=0)
    # A multiplier on a kept row is the base's multiplier on that row, zero elsewhere.
    checked = 0
    for j in range(keep.size):
        lam = np.zeros(keep.size)
        lam[j] = 1.7
        full = np.zeros(m)
        full[keep[j]] = 1.7
        np.testing.assert_array_equal(
            sub.evaluate_hessian_values(x, 0.3, lam), ev.evaluate_hessian_values(x, 0.3, full)
        )
        checked += 1
    assert checked == keep.size > 0
    # the source constraint list would describe the full row set: refused, not forwarded
    with pytest.raises(TypeError):
        sub._source_constraints
