"""#1978: the Python spatial tree's branching score (``DISCOPT_PY_TREE_SPATIAL_SCORE``).

The Python tree bisected the widest continuous column relative to its root range,
across every continuous column -- including ones a model uses only linearly, whose
split shrinks no envelope. On #1975's rescaled prolog outer piece (a bounded 21-column
problem whose margin is only resolved by splitting the narrow factors ``u0, u1``
against their wide partners ``x2..x5``) the bound stayed at 0 for thousands of nodes.
ON, the tree scores each column by its share of the model's term gaps at the node LP
point (#1970's native rule, ported) and the bound lifts within a few dozen nodes.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.mccormick_lp import MccormickLPRelaxer

FLAG = "DISCOPT_PY_TREE_SPATIAL_SCORE"


def _prolog_outer_piece(M: float = 1e4) -> dm.Model:
    """prolog's outer piece ``x7 + x8 >= M`` rescaled by ``s = 1/T`` (#1975 section 90)."""
    m = dm.Model("prolog_outer")
    s = m.continuous("s", lb=0.0, ub=1.0 / M)
    u = {k: m.continuous(f"u{k}", lb=0.0) for k in (0, 1, 6, 7, 8, 18, 19)}
    x = {k: m.continuous(f"x{k}", lb=0.0) for k in [2, 3, 4, 5, *range(9, 18)]}
    c = m.subject_to
    c(u[7] + u[8] == 1)
    for k in (0, 1, 6):
        c(u[k] >= 0.2 * s)
    c(x[2] * u[0] + x[4] * u[1] - u[18] <= 0)
    c(x[3] * u[0] + x[5] * u[1] - u[19] <= 0)
    s1, s2, s3 = (sum(x[k] for k in r) for r in ((9, 10, 11), (12, 13, 14), (15, 16, 17)))
    c(x[2] + x[3] - 0.94 * s1 + 0.244 * s3 <= 0)
    c(0.064 * s1 - 0.58 * s2 + 0.172 * s3 <= 0)
    c(x[4] + x[5] + 0.048 * s1 + 0.247 * s2 - 0.916 * s3 <= 0)
    xs = [x[k] for k in range(9, 18)]
    c(sum(a * v for a, v in zip([1, 1.2, 0.8, 2, 1.8, 2.4, 3, 2.7, 3.2], xs)) <= 3712)
    c(sum(a * v for a, v in zip([2, 1.8, 2.2, 3, 3.5, 2.3, 3, 3.2, 2.7], xs)) <= 5000)
    c(
        356.474947137148 * u[0] + 53.7083537310174 * u[1] + s * x[2] - 0.564264890180399 * u[18]
        <= 352 * s
    )
    c(
        339.983422262764 * u[0] + 43.5418249774113 * u[1] + s * x[3] - 0.405939876920766 * u[19]
        <= 430 * s
    )
    c(
        106.946746119538 * u[0] + 145.018955433089 * u[1] + s * x[4] - 0.507117039797071 * u[18]
        <= 222 * s
    )
    c(
        173.929713444361 * u[0] + 203.031384299627 * u[1] + s * x[5] - 0.578889145413521 * u[19]
        <= 292 * s
    )
    c(u[18] - 3340.8 * u[7] - 500 * u[8] <= 0)
    c(u[19] - 371.2 * u[7] - 4500 * u[8] <= 0)
    for c0, c1, c6, rows in [
        (0.94, -0.048, -0.064, [(1, 2), (1.2, 1.8), (0.8, 2.2)]),
        (0.0, -0.247, 0.58, [(2, 3), (1.8, 3.5), (2.4, 2.3)]),
        (-0.244, 0.916, -0.172, [(3, 3), (2.7, 3.2), (3.2, 2.7)]),
    ]:
        for c7, c8 in rows:
            c(c0 * u[0] + c1 * u[1] + c6 * u[6] - c7 * u[7] - c8 * u[8] <= 0)
    m.minimize(-(x[2] * u[0] + x[3] * u[0] + x[4] * u[1] + x[5] * u[1]) + 3712 * u[7] + 5000 * u[8])
    return m


@pytest.mark.unit
def test_score_terms_are_the_models_own_products():
    """The registered terms are the model's bilinear products, NOT the engine's
    ``bilinear_map`` -- which also carries the products RLT lifts (#1944). Scoring
    those steered branching onto ``u18``/``u19``, columns the model uses linearly."""
    m = _prolog_outer_piece()
    names = [v.name for v in m._variables]
    pairs, monos = MccormickLPRelaxer(m)._score_terms
    got = {tuple(sorted((names[a], names[b]))) for a, b in pairs}
    want = {
        ("s", "x2"),
        ("s", "x3"),
        ("s", "x4"),
        ("s", "x5"),
        ("u0", "x2"),
        ("u0", "x3"),
        ("u1", "x4"),
        ("u1", "x5"),
    }
    assert got == want, got
    assert monos == []


def _solve(monkeypatch, arm: str, max_nodes: int):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")  # the Python tree
    monkeypatch.setenv(FLAG, arm)
    return _prolog_outer_piece().solve(
        time_limit=60, gap_tolerance=1e-4, abs_gap_tolerance=1e-3, max_nodes=max_nodes
    )


@pytest.mark.slow
@pytest.mark.correctness
def test_flag_off_python_tree_stalls_at_zero(monkeypatch):
    """The control: the legacy rule's bound stays at ~0 (the true value is ~0.0349)."""
    r = _solve(monkeypatch, "0", max_nodes=150)
    assert r.bound is not None and r.bound < 1e-6, r.bound
    assert not r.gap_certified


@pytest.mark.slow
@pytest.mark.correctness
def test_flag_on_python_tree_lifts_the_bound_and_certifies(monkeypatch):
    r = _solve(monkeypatch, "1", max_nodes=150)
    picks = (r.solver_stats or {}).get("tree/py_spatial_score_picks", 0)
    assert picks > 0, "the score rule never fired: the probe measured nothing"
    # 0.034950 is the native kernel's certified value for this piece (#1978).
    assert r.bound is not None and 0.0345 < r.bound <= 0.034951, r.bound
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound, r.objective)
