"""Issue #1867, part 2: integer cross-term blocks (``ball_mk4_*``) and the convex route's box
(``ball_mk2_*``).

``ball_mk4_*`` is one row ``sum_pairs (100x^2 + 100y^2 - 98x - 98y - 4xy) + 1 <= 0`` over
integers ``x >= -100``. Each pair reaches ``-49`` at ``(0.5, 0.5)`` but equals
``98(x^2 - x) + 98(y^2 - y) + 2(x - y)^2 >= 0`` on Z^2, so the row is infeasible.
:class:`IntegerBlockQuadraticRule` proves it from the integer minimum of each block.

``ball_mk2_*``'s box is pinned to ``{0}`` by the part-1 integer minimum, but the convex
MINLP route was handed the *declared* box and ran outer approximation on it.
``DISCOPT_ROUTE_TIGHTENED_BOX`` hands the route the tightened box instead.

Each claim is checked against an independent oracle: brute-force enumeration, exact
integer arithmetic, and feasible-point sampling.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from discopt._relax import nonlinear_bound_tightening as N
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

BLOCK_FLAG = "DISCOPT_INTEGER_BLOCK_QUAD"
BOX_FLAG = "DISCOPT_ROUTE_TIGHTENED_BOX"


def _mk4(pairs: int, rhs: float, lb: float = -100.0) -> Model:
    """The ``ball_mk4`` row, spelled as MINLPLib writes it (there ``ub = inf``; an
    explicit large ``ub`` here keeps the default-integer-box warning out of the test
    without changing the proof, whose ellipsoid never reaches it)."""
    m = Model("mk4")
    x = [m.integer(f"x{i}", lb=lb, ub=1e4) for i in range(2 * pairs)]
    body = 0
    for i in range(0, 2 * pairs, 2):
        a, b = x[i], x[i + 1]
        body = body + 100 * a**2 - 98 * a + 100 * b**2 - 98 * b - 2 * a * b - 2 * a * b
    m.subject_to(body <= rhs)
    m.minimize(sum((i + 1) * v for i, v in enumerate(x)))
    return m


def _tighten(m: Model):
    meta = N.build_flat_variable_metadata(m)
    from discopt.solver import flat_variable_bounds

    lb, ub = flat_variable_bounds(m)
    return N.IntegerBlockQuadraticRule().tighten(m, lb.copy(), ub.copy(), meta)


def _q(Q, c, x):
    return float(x @ Q @ x + c @ x)


def test_block_minimum_matches_brute_force_and_never_undercuts_the_continuous_one():
    rng = np.random.default_rng(1867)
    checked = 0
    for _ in range(300):
        k = int(rng.integers(2, 4))
        A = rng.normal(size=(k, k))
        Q = A @ A.T + 0.05 * np.eye(k)
        c = rng.normal(scale=4.0, size=k)
        lo = rng.integers(-4, 2, size=k).astype(float) - rng.uniform(0, 0.9, size=k)
        hi = lo + rng.integers(0, 6, size=k) + rng.uniform(0, 0.9, size=k)
        got, _ = N._min_quadratic_block_integer(Q, c, lo, hi)
        axes = [np.arange(np.ceil(a), np.floor(b) + 1) for a, b in zip(lo, hi)]
        pts = [np.array(p, dtype=float) for p in itertools.product(*axes)]
        if not pts:
            continue
        brute = min(_q(Q, c, p) for p in pts)
        cont = float(-0.25 * c @ np.linalg.solve(Q, c))
        grad = max(np.abs(2 * Q @ p + c).sum() for p in pts)
        # A lower bound on the integer minimum (within the tolerance window)...
        assert got <= brute + 1e-9 * (1 + abs(brute)), (Q, c, lo, hi, got, brute)
        assert got >= brute - grad * N._INTEGER_DOMAIN_TOL - 1e-9 * (1 + abs(brute))
        # ...and never below the continuous minimum.
        assert got >= cont - 1e-9 * (1 + abs(cont))
        checked += 1
    assert checked > 200, checked


def test_the_mk4_pair_is_nonnegative_on_the_integers_exactly():
    """The identity the infeasibility proof rests on, in exact integer arithmetic."""
    checked = 0
    for x in range(-100, 101):
        for y in range(-100, 101):
            q = 100 * x * x + 100 * y * y - 98 * x - 98 * y - 4 * x * y
            assert q == 98 * (x * x - x) + 98 * (y * y - y) + 2 * (x - y) ** 2
            assert q >= 0
            checked += 1
    assert checked == 201 * 201


@pytest.mark.parametrize("pairs", [5, 10, 15])
def test_flag_on_proves_the_mk4_shape_infeasible(pairs, monkeypatch):
    monkeypatch.setenv(BLOCK_FLAG, "1")
    with pytest.raises(N.NonlinearBoundTighteningInfeasible):
        _tighten(_mk4(pairs, -1.0))


def test_flag_off_proves_nothing(monkeypatch):
    monkeypatch.setenv(BLOCK_FLAG, "0")
    _tighten(_mk4(5, -1.0))  # must not raise


def test_a_feasible_row_is_never_proved_infeasible(monkeypatch):
    """Feasible-point sampling: ``rhs = 0`` is satisfied at 0 and at (1, 1) pairs; the
    rule must not raise, and every sampled point it would have to cut is feasible."""
    monkeypatch.setenv(BLOCK_FLAG, "1")
    m = _mk4(3, 0.0)
    _tighten(m)  # must not raise
    rng = np.random.default_rng(3)
    feasible = 0
    for _ in range(5000):
        x = rng.integers(-1, 3, size=6)
        q = sum(
            100 * a * a + 100 * b * b - 98 * a - 98 * b - 4 * a * b for a, b in zip(x[::2], x[1::2])
        )
        feasible += int(q <= 0)
    assert feasible > 0, "sampling found no feasible point -- the check measured nothing"


def test_a_non_positive_definite_block_abstains(monkeypatch):
    """``x*y`` with no squares is indefinite: the rule must skip the row, not bound it."""
    monkeypatch.setenv(BLOCK_FLAG, "1")
    m = Model("indef")
    x = m.integer("x", lb=-3, ub=3)
    y = m.integer("y", lb=-3, ub=3)
    m.subject_to(x * y <= -100)  # infeasible on the box, but not this rule's to prove
    m.minimize(x + y)
    meta = N.build_flat_variable_metadata(m)
    assert N.IntegerBlockQuadraticRule()._row_blocks(m, m._constraints[0], meta) is None


def test_box_edge_within_the_integrality_tolerance_keeps_the_nearby_integer():
    """``x = 1 + 5e-6`` is accepted as integral 1; the block minimum must not exceed
    the value there."""
    Q = np.array([[1.0, 0.2], [0.2, 1.0]])
    c = -2.0 * Q @ np.array([1.0, 1.0])  # minimum at (1, 1)
    lb = np.array([1.0 + 5e-6, 1.0 + 5e-6])
    ub = np.array([3.0, 3.0])
    got, _ = N._min_quadratic_block_integer(Q, c, lb, ub)
    assert got <= _q(Q, c, lb) + 1e-12


def test_integer_block_flag_defaults_on(monkeypatch):
    monkeypatch.delenv(BLOCK_FLAG, raising=False)
    assert N.integer_block_quadratic_enabled()
    monkeypatch.setenv(BLOCK_FLAG, "0")
    assert not N.integer_block_quadratic_enabled()


# ── the convex route's box (ball_mk2) ──


def _mk2(n: int = 10) -> Model:
    m = Model("mk2")
    x = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.subject_to(sum(xi**2 - 0.987 * xi for xi in x) <= 0)
    m.minimize(-sum(x))
    return m


def _route_box(monkeypatch, flag: str):
    """Solve ``_mk2`` and capture the box the MIP-NLP route was handed."""
    import discopt.solvers.mip_nlp as mip_nlp

    monkeypatch.setenv(BOX_FLAG, flag)
    real = mip_nlp.solve_mip_nlp
    seen = []

    def spy(model, *a, **kw):
        seen.append([(v.lb.copy(), v.ub.copy()) for v in model._variables])
        return real(model, *a, **kw)

    monkeypatch.setattr(mip_nlp, "solve_mip_nlp", spy)
    m = _mk2()
    declared = [(v.lb.copy(), v.ub.copy()) for v in m._variables]
    r = m.solve(time_limit=20)
    assert seen, "the convex route did not run -- the probe measured nothing"
    # The caller's model is never modified.
    for (lb, ub), v in zip(declared, m._variables):
        assert np.array_equal(lb, v.lb) and np.array_equal(ub, v.ub)
    return r, seen[0]


def test_route_receives_the_tightened_box(monkeypatch):
    r, box = _route_box(monkeypatch, "1")
    assert all(float(lb) == 0.0 and float(ub) == 0.0 for lb, ub in box), box
    assert r.status == "optimal" and r.objective == pytest.approx(0.0)
    assert (r.algorithm_route or "").startswith("mip-nlp/")


def test_route_box_flag_off_is_the_declared_box(monkeypatch):
    _, box = _route_box(monkeypatch, "0")
    assert all(float(lb) == -1.0 and float(ub) == 1.0 for lb, ub in box), box


def test_route_box_flag_defaults_on(monkeypatch):
    from discopt.solver import _route_tightened_box_enabled

    monkeypatch.delenv(BOX_FLAG, raising=False)
    assert _route_tightened_box_enabled()
    monkeypatch.setenv(BOX_FLAG, "0")
    assert not _route_tightened_box_enabled()
