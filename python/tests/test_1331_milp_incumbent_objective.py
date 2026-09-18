"""#1331: a MI(L|Q)P exit must report the objective ITS OWN x achieves.

``_solve_milp_bb`` stored the objective its *node relaxation* reported, then
integer-snapped the point (the C-3 snap) on the way out and returned the stale
value. The snap moves the objective by ``sum_j |c_j| |dx_j|``, so a 14-variable
integer knapsack came back

    optimal, objective = bound = 180.00004616, gap_certified=True

at a point achieving exactly 180 -- an incumbent value no point attains, and
one BETTER than the true optimum, overstated by ~4.6e4 times the requested
absolute tolerance. ``_solve_miqp_bb``'s exit has the same shape and the same
snap, so it is fixed and covered here too.

The fix recomputes the objective from the (snapped, feasibility-verified) point
that is returned. When that recomputation makes the value WORSE, the tree
fathomed and stopped against a value nothing attains, so the convergence test is
re-run against the honest (incumbent, bound) pair before granting ``optimal``.
"""

import discopt.modeling as dm
import numpy as np
import pytest


def _knapsack(seed, sense="max", offset=0.0, name=None):
    """An integer knapsack in the family the issue reproduced on."""
    rng = np.random.default_rng(seed)
    n = 14
    p = rng.integers(3, 30, size=n).astype(float)
    w = rng.integers(2, 20, size=n).astype(float)
    cap = float(np.round(0.45 * 2 * w.sum()))

    m = dm.Model(name or f"i1331_knap_{sense}_{seed}")
    y = m.integer("y", shape=(n,), lb=0, ub=2)
    m.subject_to(sum(w[i] * y[i] for i in range(n)) <= cap)
    body = sum(p[i] * y[i] for i in range(n)) + offset
    if sense == "max":
        m.maximize(body)
    else:
        m.minimize(-body)
    return m, p, offset


def _solution_vector(result):
    values = list(result.x.values()) if isinstance(result.x, dict) else [result.x]
    return np.asarray(values[0], dtype=np.float64).ravel()


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_milp_objective_is_achieved_by_the_returned_point(seed, monkeypatch):
    """``objective == c @ x`` exactly. Pre-fix this was off by up to 2.6e-5.

    ``abs_gap_tolerance`` below the relative default is what forces the Python
    tree, per the issue.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m, p, _ = _knapsack(seed)
    r = m.solve(abs_gap_tolerance=1e-9)

    assert r.objective is not None, "the knapsack is feasible; an incumbent is expected"
    y = _solution_vector(r)
    assert np.allclose(y, np.round(y), atol=1e-6), "an integer variable came back fractional"
    achieved = float(p @ np.round(y))
    assert r.objective == pytest.approx(achieved, abs=1e-9), (
        f"reported objective {r.objective!r} is not achieved by the returned point ({achieved!r})"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [0, 2])
def test_milp_certificate_brackets_the_reported_objective(seed, monkeypatch):
    """A maximize bound is an UPPER bound on a value the point actually attains."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m, _p, _ = _knapsack(seed)
    r = m.solve(abs_gap_tolerance=1e-9)
    if r.bound is None:
        pytest.skip("no dual bound reported on this exit")
    assert r.bound >= r.objective - 1e-6, (
        f"bound {r.bound!r} is below an attained objective {r.objective!r}"
    )


@pytest.mark.correctness
def test_milp_objective_includes_the_constant_term(monkeypatch):
    """The recomputation must use ``obj_const``, not just ``c @ x``."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    offset = 137.5
    m, p, _ = _knapsack(0, offset=offset, name="i1331_offset")
    r = m.solve(abs_gap_tolerance=1e-9)
    y = np.round(_solution_vector(r))
    assert r.objective == pytest.approx(float(p @ y) + offset, abs=1e-9)


@pytest.mark.correctness
def test_milp_minimize_sense_is_unchanged(monkeypatch):
    """The control: the minimize path must report its own point's value too."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m, p, _ = _knapsack(1, sense="min", name="i1331_min")
    r = m.solve(abs_gap_tolerance=1e-9)
    y = np.round(_solution_vector(r))
    assert r.objective == pytest.approx(float(-(p @ y)), abs=1e-9)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6


@pytest.mark.correctness
def test_miqp_objective_is_achieved_by_the_returned_point(monkeypatch):
    """The MIQP exit shares the snap and the stale value; it shares the fix."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    rng = np.random.default_rng(7)
    n = 8
    target = rng.uniform(0.0, 3.0, size=n)

    m = dm.Model("i1331_miqp")
    y = m.integer("y", shape=(n,), lb=0, ub=3)
    m.subject_to(sum(y[i] for i in range(n)) <= 12)
    m.minimize(sum((y[i] - float(target[i])) ** 2 for i in range(n)))
    r = m.solve(abs_gap_tolerance=1e-9)

    assert r.objective is not None
    yv = np.round(_solution_vector(r))
    achieved = float(np.sum((yv - target) ** 2))
    assert r.objective == pytest.approx(achieved, abs=1e-7), (
        f"MIQP reported {r.objective!r} at a point achieving {achieved!r}"
    )
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6


# ── the recomputation itself ────────────────────────────────────────────────


@pytest.mark.unit
def test_objective_at_reported_point_linear_and_quadratic():
    from discopt.solver import _objective_at_reported_point

    x = np.array([1.0, 2.0, 3.0])
    c = np.array([2.0, -1.0, 0.5])
    assert _objective_at_reported_point(x, c, 0.0) == pytest.approx(1.5)
    assert _objective_at_reported_point(x, c, 10.0) == pytest.approx(11.5)

    Q = np.diag([2.0, 2.0, 2.0])
    # 0.5 x'Qx = |x|^2 = 14
    assert _objective_at_reported_point(x, c, 0.0, Q=Q) == pytest.approx(15.5)
    assert _objective_at_reported_point(x, np.zeros(3), 1.0, Q=np.zeros((3, 3))) == pytest.approx(
        1.0
    )


@pytest.mark.unit
def test_objective_at_reported_point_matches_a_snapped_point_not_the_node_value():
    """The arithmetic of the defect: a 1e-6 snap on 14 terms moves the value."""
    from discopt.solver import _objective_at_reported_point

    rng = np.random.default_rng(3)
    c = rng.uniform(3.0, 30.0, size=14)
    x_int = rng.integers(0, 3, size=14).astype(float)
    x_node = x_int + rng.uniform(-1e-6, 1e-6, size=14)

    at_node = _objective_at_reported_point(x_node, c, 0.0)
    at_snapped = _objective_at_reported_point(x_int, c, 0.0)
    assert at_node != at_snapped, "the family must actually separate the two values"
    assert at_snapped == pytest.approx(float(c @ x_int), abs=0.0)
