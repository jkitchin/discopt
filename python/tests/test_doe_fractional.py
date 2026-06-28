"""Tests for fractional factorial designs via MILP row selection."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from discopt.doe import (
    FactorialDesign,
    effects_estimates,
    fractional_factorial_design,
)


def _coded(design: FactorialDesign) -> np.ndarray:
    """Return rows as a coded ±1 matrix in factor order."""
    rows = [r for r in design.rows if not r.get("is_center", False)]
    n = len(rows)
    k = len(design.factors)
    M = np.zeros((n, k), dtype=int)
    for ri, row in enumerate(rows):
        for ci, name in enumerate(design.factors):
            M[ri, ci] = 1 if row[name] == design.high[ci] else -1
    return M


def test_half_fraction_2pow4_resolution4():
    """Half-fraction of 2^4 at resolution IV: 8 runs, all 2FIs balanced."""
    factors = {n: (-1, 1) for n in ["A", "B", "C", "D"]}
    d = fractional_factorial_design(factors, n_runs=8, resolution=4, seed=0)
    M = _coded(d)
    assert M.shape == (8, 4)
    # Main effects balanced and pairwise orthogonal.
    assert np.array_equal(M.sum(axis=0), np.zeros(4, dtype=int))
    assert np.array_equal(M.T @ M, 8 * np.eye(4, dtype=int))
    # Every 3FI column balanced (R=IV signature).
    for i, j, a in itertools.combinations(range(4), 3):
        assert (M[:, i] * M[:, j] * M[:, a]).sum() == 0


def test_quarter_fraction_2pow5_resolution3():
    """Quarter-fraction of 2^5 at resolution III: 8 runs, mains orthogonal."""
    factors = {f"F{i}": (-1, 1) for i in range(5)}
    d = fractional_factorial_design(factors, n_runs=8, resolution=3, seed=0)
    M = _coded(d)
    assert M.shape == (8, 5)
    assert np.array_equal(M.sum(axis=0), np.zeros(5, dtype=int))
    assert np.array_equal(M.T @ M, 8 * np.eye(5, dtype=int))


def test_2pow5_resolution5_full_clearance():
    """Resolution V on k=5: 16 runs, all main+2FI columns mutually orthogonal."""
    factors = {f"F{i}": (-1, 1) for i in range(5)}
    d = fractional_factorial_design(factors, n_runs=16, resolution=5, seed=0)
    M = _coded(d)
    assert M.shape == (16, 5)
    twofis = np.array([M[:, i] * M[:, j] for i, j in itertools.combinations(range(5), 2)]).T
    # mains balanced.
    assert np.array_equal(M.sum(axis=0), np.zeros(5, dtype=int))
    # mains ⊥ 2FIs.
    assert np.all(M.T @ twofis == 0)
    # 2FIs ⊥ each other (off-diagonal of Gram is zero).
    G = twofis.T @ twofis
    assert np.all(G - np.diag(np.diag(G)) == 0)


@pytest.mark.slow
def test_k7_resolution4_breaks_8_factor_cap():
    """k=7 at R=IV: 16 runs — beyond the full-factorial implementation's k<=8 cap.

    Selecting 16 of the 2^7=128 candidate rows is a combinatorial orthogonality
    MILP (`_solve_row_milp`) that takes ~70s locally and exceeds the 120s PR-fast
    budget on the slower CI runner, so it lives in the slow tier rather than the
    per-commit suite. Not a regression — timing is identical on main.
    """
    factors = {f"X{i}": (0.0, 1.0) for i in range(7)}
    d = fractional_factorial_design(factors, n_runs=16, resolution=4, seed=1)
    M = _coded(d)
    assert M.shape == (16, 7)
    assert np.array_equal(M.sum(axis=0), np.zeros(7, dtype=int))
    assert np.array_equal(M.T @ M, 16 * np.eye(7, dtype=int))


def test_recovers_main_effects_on_linear_response():
    """OLS recovers main effects correctly from a fractional design."""
    rng = np.random.default_rng(0)
    factors = {n: (-1.0, 1.0) for n in ["A", "B", "C", "D"]}
    d = fractional_factorial_design(factors, n_runs=8, resolution=4, seed=42)
    # True model: y = 1 + 2*A - 3*B + 0.5*C + noise (no D effect, no interactions).
    true_effects = {"A": 2.0, "B": -3.0, "C": 0.5, "D": 0.0}
    for row in d.rows:
        if row.get("is_center"):
            continue
        y = 1.0
        for name, beta in true_effects.items():
            y += beta * float(row[name])
        y += float(rng.normal(0, 0.01))
        row["y"] = y

    est = effects_estimates(d.rows, response="y")
    by_factor = {e["factor"]: e["effect"] for e in est}
    # effect = 2 * beta for ±1 coding.
    assert by_factor["A"] == pytest.approx(4.0, abs=0.1)
    assert by_factor["B"] == pytest.approx(-6.0, abs=0.1)
    assert by_factor["C"] == pytest.approx(1.0, abs=0.1)
    assert by_factor["D"] == pytest.approx(0.0, abs=0.1)


def test_categorical_factors_round_trip():
    """Categorical low/high values appear in the output rows."""
    factors = {
        "catalyst": ("A", "B"),
        "temp": (80.0, 120.0),
        "solvent": ("EtOH", "MeOH"),
        "pressure": (1.0, 5.0),
    }
    d = fractional_factorial_design(factors, n_runs=8, resolution=3, seed=0)
    levels_seen: dict[str, set] = {n: set() for n in factors}
    for row in d.rows:
        for name in factors:
            levels_seen[name].add(row[name])
    for name, (lo, hi) in factors.items():
        assert levels_seen[name] == {lo, hi}


def test_replicates_multiply_row_count():
    factors = {n: (-1, 1) for n in ["A", "B", "C"]}
    d = fractional_factorial_design(factors, n_runs=4, resolution=3, replicates=3, seed=0)
    rows = [r for r in d.rows if not r.get("is_center", False)]
    assert len(rows) == 12
    reps = sorted({r["replicate"] for r in rows})
    assert reps == [0, 1, 2]


def test_center_points_added_per_replicate():
    factors = {n: (-1.0, 1.0) for n in ["A", "B", "C", "D"]}
    d = fractional_factorial_design(
        factors, n_runs=8, resolution=4, center_points=2, replicates=2, seed=0
    )
    centers = [r for r in d.rows if r.get("is_center")]
    assert len(centers) == 4
    for c in centers:
        for name in factors:
            assert c[name] == 0.0


def test_center_points_rejected_for_categorical():
    factors = {"catalyst": ("A", "B"), "temp": (80.0, 120.0), "x": (0.0, 1.0), "y": (0.0, 1.0)}
    with pytest.raises(ValueError, match="non-numeric"):
        fractional_factorial_design(factors, n_runs=8, resolution=3, center_points=1)


def test_n_runs_must_be_power_of_2():
    factors = {n: (-1, 1) for n in ["A", "B", "C", "D"]}
    with pytest.raises(ValueError, match="power of 2"):
        fractional_factorial_design(factors, n_runs=6, resolution=3)


def test_resolution_validation():
    factors = {n: (-1, 1) for n in ["A", "B", "C"]}
    with pytest.raises(ValueError, match="resolution"):
        fractional_factorial_design(factors, n_runs=4, resolution=6)


def test_infeasible_too_few_runs_raises():
    # R=5 with k=5 requires 16+. Asking for 8 (the minimum power of 2
    # below the R=5 floor of 16) should be rejected up front.
    factors = {f"F{i}": (-1, 1) for i in range(5)}
    with pytest.raises(ValueError, match="too small"):
        fractional_factorial_design(factors, n_runs=8, resolution=5)


def test_runs_order_is_shuffled_deterministically():
    factors = {n: (-1, 1) for n in ["A", "B", "C", "D"]}
    d1 = fractional_factorial_design(factors, n_runs=8, resolution=4, seed=7)
    d2 = fractional_factorial_design(factors, n_runs=8, resolution=4, seed=7)
    d3 = fractional_factorial_design(factors, n_runs=8, resolution=4, seed=8)
    sig1 = [tuple(r[f] for f in d1.factors) for r in d1.rows]
    sig2 = [tuple(r[f] for f in d2.factors) for r in d2.rows]
    sig3 = [tuple(r[f] for f in d3.factors) for r in d3.rows]
    assert sig1 == sig2
    # Different seed should usually permute (sets equal, list possibly differs).
    assert sorted(sig1) == sorted(sig3)
