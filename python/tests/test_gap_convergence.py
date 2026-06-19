"""Unit tests for the B&B gap-convergence criterion (``_gap_converged``).

The integer B&B previously certified optimality on ``tree.gap() <= gap_tolerance``,
where ``tree.gap()`` floors its denominator at 1.0 — so for an optimum with
magnitude below 1 the relative ``gap_tolerance`` silently became an *absolute*
tolerance. On ``gear`` (objective is a squared residual, relaxation bound ≈ 0) a
coarse 1e-4 certified ``obj=3.7e-05`` at node 3, ~7 orders of magnitude from the
true ~4e-12 optimum. ``_gap_converged`` decouples the two: converge on
``relative_gap <= gap_tolerance`` OR ``absolute_gap <= abs_gap_tol`` (default 1e-6,
matching the AMP path), with the relative gap computed without the 1.0 floor.
"""

from __future__ import annotations

from discopt.solver import _DEFAULT_ABS_GAP_TOL, _gap_converged


class _FakeTree:
    """Minimal stand-in exposing the ``stats()`` dict ``_gap_converged`` reads."""

    def __init__(self, ub: float, lb: float):
        self._ub = ub
        self._lb = lb

    def stats(self):
        return {"incumbent_value": self._ub, "global_lower_bound": self._lb}


def test_default_abs_tol_matches_amp_path():
    assert _DEFAULT_ABS_GAP_TOL == 1e-6


def test_near_zero_optimum_tiny_absolute_gap_converges():
    """gear after the fix: UB=1.8e-08, LB≈0 → absolute gap below 1e-6 → certified."""
    assert _gap_converged(_FakeTree(1.827e-08, 0.0), gap_tolerance=1e-4)


def test_near_zero_optimum_loose_absolute_gap_does_not_converge():
    """The gear pathology: UB=3.7e-05 on a trivial 0 bound. Relative gap is ~100%
    and the absolute gap (3.7e-05) is above 1e-6 — must NOT certify (it used to,
    because the floored-denominator gap made 1e-4 an absolute tolerance)."""
    assert not _gap_converged(_FakeTree(3.69e-05, 0.0), gap_tolerance=1e-4)


def test_normal_optimum_relative_convergence_unchanged():
    """An O(10) optimum with a 1e-5 relative gap still certifies via the relative
    branch (absolute gap 1e-4 is above abs_gap_tol, so relative carries it)."""
    assert _gap_converged(_FakeTree(10.0001, 10.0), gap_tolerance=1e-4)


def test_normal_optimum_open_gap_does_not_converge():
    assert not _gap_converged(_FakeTree(12.0, 10.0), gap_tolerance=1e-4)


def test_sub_one_optimum_uses_relative_not_floored_absolute():
    """Optimum 0.5 with a 40% true relative gap must NOT certify at gap_tolerance
    1e-4 — the old floored-denominator gap reported 0.4/1.0=40% as an absolute
    0.4 and (incorrectly, for relative purposes) compared it the same way; the
    decoupled relative gap is 0.4/0.5 = 80% >> 1e-4."""
    assert not _gap_converged(_FakeTree(0.5, 0.1), gap_tolerance=1e-4)


def test_absolute_gap_certifies_genuinely_zero_optimum():
    """A true-zero optimum (sum of squares): relative gap is undefined/huge, so the
    independent absolute tolerance is the only sound certificate."""
    assert _gap_converged(_FakeTree(5e-7, 0.0), gap_tolerance=1e-4)
    assert not _gap_converged(_FakeTree(5e-5, 0.0), gap_tolerance=1e-4)


def test_no_bound_does_not_converge():
    assert not _gap_converged(_FakeTree(float("inf"), float("-inf")), gap_tolerance=1e-4)
    assert not _gap_converged(_FakeTree(1.0, float("-inf")), gap_tolerance=1e-4)


def test_explicit_abs_tol_override():
    tree = _FakeTree(3.69e-05, 0.0)
    assert not _gap_converged(tree, gap_tolerance=1e-4, abs_gap_tol=1e-6)
    assert _gap_converged(tree, gap_tolerance=1e-4, abs_gap_tol=1e-4)
