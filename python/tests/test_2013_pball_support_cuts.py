"""#2013 (p_ball): why the routed ``lp_nlp_bb`` master's bound stayed at ~0, and the
two fixes.

p_ball assigns points to unit balls through big-M rows
``||x_p - c_b||^2 + M y_pb <= M + 1``. Two mechanisms, measured on
p_ball_10b_5p_4d_m (30 s, HiGHS master):

1. **The master had no row that bounds ``x_p`` once ``y_pb = 1``.** Linearised at
   the relaxation point alone, every integer master point put all points on top of
   each other at objective 0; the tree was killed at that first incumbent and every
   one of 11 trees ended with dual bound 0.0. ``DISCOPT_OA_SUPPORT_CUTS`` seeds the
   master with the tangents at the extremes of each nonlinear variable of each
   convex row (for a ball at ``y = 1``: its four axis-aligned supporting planes),
   after which the trees end at 55.9 -> 62.5 (incumbent 71.37) and
   p_ball_10b_5p_2d_m certifies in 11.6 s where it was uncertified at 30 s.
2. **The in-tree check-in hid that progress.** Fixed separately and pinned by
   ``test_2013_master_checkin_running_bound.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

highspy = pytest.importorskip("highspy", reason="the HiGHS master is a core dependency")

from discopt.solvers.oa import (  # noqa: E402
    _decompose_model,
    _support_points_for_rows,
    _SupportPointReport,
    solve_lp_nlp_bb,
)


def _two_point_ball_model(dim: int = 2):
    """Two points, three unit balls, each point in exactly one ball, each ball holding
    at most one point, minimise the L1 distance between the points: a p_ball in
    miniature. Ball centres (0,0), (6,0), (0,6); optimum 6 - 2 = 4 (x-order broken by
    a symmetry row as in the MINLPLib model)."""
    import discopt.modeling as dm

    centres = [np.zeros(dim), np.r_[6.0, np.zeros(dim - 1)], np.r_[0.0, 6.0, np.zeros(dim - 2)]]
    big_m = 2.0 * dim * 10.0**2
    m = dm.Model("pball_mini")
    xs = [[m.continuous(f"x{p}_{d}", lb=0.0, ub=10.0) for d in range(dim)] for p in range(2)]
    ys = [[m.binary(f"y{p}_{b}") for b in range(3)] for p in range(2)]
    ds = [m.continuous(f"d{d}", lb=0.0, ub=10.0) for d in range(dim)]
    for p in range(2):
        for b, c in enumerate(centres):
            m.subject_to(
                sum((float(c[d]) - xs[p][d]) ** 2 for d in range(dim)) + big_m * ys[p][b]
                <= big_m + 1.0
            )
        m.subject_to(sum(ys[p]) == 1)
    for b in range(3):
        m.subject_to(ys[0][b] + ys[1][b] <= 1)
    for d in range(dim):
        m.subject_to(xs[0][d] - xs[1][d] <= ds[d])
        m.subject_to(xs[1][d] - xs[0][d] <= ds[d])
    m.subject_to(xs[0][0] <= xs[1][0])
    m.minimize(sum(ds))
    return m


def _expected_points(dim: int) -> int:
    """Boundary points over both points' rows: per centre, one per axis direction
    along which the unit ball's extreme still lies inside the box ``[0, 10]``."""
    centres = [np.zeros(dim), np.r_[6.0, np.zeros(dim - 1)], np.r_[0.0, 6.0, np.zeros(dim - 2)]]
    per_centre = [
        sum(int(c[d] + 1.0 <= 10.0) + int(c[d] - 1.0 >= 0.0) for d in range(dim)) for c in centres
    ]
    return 2 * sum(per_centre)


@pytest.mark.smoke
def test_support_points_are_the_axis_extremes_of_the_tight_slice():
    """Mechanism 1, the probe: for a ball row ``||x - c||^2 + M y <= M + 1`` the
    tight slice is ``y = 1`` and the points are ``c +/- e_d``; the tangent there is
    the axis-aligned supporting plane, so with ``y = 1`` it reads ``x_d <= c_d + 1``
    (or ``>=``)."""
    model = _two_point_ball_model()
    decomp = _decompose_model(model)
    ev = decomp.evaluator
    report = _SupportPointReport()
    points = _support_points_for_rows(
        ev,
        decomp.lb,
        decomp.ub,
        decomp.constraint_senses,
        decomp.oa_cut_mask,
        np.full(decomp.n_vars, 5.0),
        budget_s=10.0,
        report=report,
    )
    ball_rows = [k for k, s in enumerate(decomp.constraint_senses) if decomp.oa_cut_mask[k]]
    assert report.rows_probed == len(ball_rows) == 6
    # A centre on the box edge has no room on that side (the box already is the
    # bound there), so the count is per centre, not 2 * dim per row.
    assert report.boundary_points == _expected_points(2)
    assert len(points) == _expected_points(2)
    assert not report.budget_exhausted
    checked = 0
    for k, z in points:
        vals = np.asarray(ev.evaluate_constraints(z))
        assert abs(vals[k]) < 1e-6, f"row {k} support point is not on the boundary: {vals[k]}"
        # the integer column of the row sits at its tightening bound (1)
        jac = np.asarray(ev.evaluate_jacobian(z))[k]
        y_cols = [j for j in decomp.int_indices if jac[j] != 0.0]
        assert y_cols and all(z[j] == 1.0 for j in y_cols)
        # the gradient is axis-aligned in the nonlinear columns: exactly one nonzero
        x_cols = [j for j in range(decomp.n_vars) if j not in decomp.int_indices and jac[j] != 0]
        assert len(x_cols) == 1
        assert abs(abs(jac[x_cols[0]]) - 2.0) < 1e-6
        checked += 1
    assert checked == _expected_points(2)


@pytest.mark.smoke
@pytest.mark.parametrize("dim", [2, 3])
def test_support_cuts_keep_the_optimum_and_are_reported(monkeypatch, dim):
    """Mechanism 1, end to end: the flag changes the master's rows, not the answer,
    and the trace says what it did (OFF: ``None``, not zero)."""
    monkeypatch.delenv("DISCOPT_OA_SUPPORT_CUTS", raising=False)
    off = solve_lp_nlp_bb(_two_point_ball_model(dim), time_limit=60.0, milp_solver="highs")
    monkeypatch.setenv("DISCOPT_OA_SUPPORT_CUTS", "1")
    on = solve_lp_nlp_bb(_two_point_ball_model(dim), time_limit=60.0, milp_solver="highs")
    assert off.status == "optimal" and on.status == "optimal"
    assert off.objective == pytest.approx(4.0, abs=1e-4)
    assert on.objective == pytest.approx(off.objective, abs=1e-4)
    assert on.bound <= on.objective + 1e-6
    off_rep = off.mip_nlp_trace["summary"]["support_cuts"]
    on_rep = on.mip_nlp_trace["summary"]["support_cuts"]
    assert off_rep is None
    assert on_rep["rows_probed"] == 6
    assert on_rep["boundary_points"] == _expected_points(dim)
    assert on.mip_nlp_trace["summary"]["cut_source_counts"]["support"] == _expected_points(dim)
    assert "support" not in off.mip_nlp_trace["summary"]["cut_source_counts"]


@pytest.mark.smoke
def test_support_probe_takes_the_model_columns_of_a_master_point(monkeypatch):
    """A cut record's point is in the master's column space, which appends the
    objective epigraph after the model's columns. The probe must hand the evaluator
    the model's columns only: on batch the unsliced point raised "expected length 46,
    got 47" and the route fell back. A convex nonlinear objective gives the master
    that extra column here."""

    def model():
        m = _two_point_ball_model(2)
        xs = [v for v in m._variables if v.name.startswith("x")]
        m.minimize(sum((x - 3.0) ** 2 for x in xs))
        return m

    monkeypatch.delenv("DISCOPT_OA_SUPPORT_CUTS", raising=False)
    off = solve_lp_nlp_bb(model(), time_limit=60.0, milp_solver="highs")
    monkeypatch.setenv("DISCOPT_OA_SUPPORT_CUTS", "1")
    on = solve_lp_nlp_bb(model(), time_limit=60.0, milp_solver="highs")
    assert off.status == "optimal" and on.status == "optimal", (off.status, on.status)
    assert on.objective == pytest.approx(off.objective, abs=1e-4)
    report = on.mip_nlp_trace["summary"]["support_cuts"]
    assert report is not None and report["boundary_points"] > 0, report
