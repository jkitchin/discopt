"""Regression tests for issue #756 — two false-infeasible certificates in GDPopt-LOA.

Both defects produced ``status="infeasible"`` with ``gap_certified=True`` on *feasible*
models (hard-gate territory per CLAUDE.md — same class as #739/#740):

1. ``_add_no_good_cut`` used the binary-only exclusion form for general integer
   variables; the cut for a proven-infeasible config also cut off feasible non-0/1
   configurations, so LOA excluded everything and certified a false infeasible.
2. Exiting the LOA loop on the time limit (or the iteration cap, or a non-infeasible
   master verdict) fell through to a certified ``"infeasible"`` — a timeout is not an
   infeasibility proof.

These solve on the in-house simplex/POUNCE master, so no highspy is required.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers import gdpopt_loa as loa


@pytest.mark.smoke
class TestIssue756NoGoodSoundness:
    def test_general_integer_no_good_cut_soundness(self):
        """#756(1): y binary, z integer in [0, 3], y*z >= 2 — optimum 3 at (1, 2).

        Fails before the fix: the binary-only no-good cut for config (0, 1) is
        ``-y + z <= 0``, which also cuts the feasible optimum (1, 2); LOA excludes
        every config and returns a certified false ``"infeasible"``.
        """
        m = dm.Model("loa_nogood_int")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=3)
        m.subject_to(y * z >= 2)
        m.minimize(y + z)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible"), (
            f"feasible model declared {r.status!r} (false infeasibility)"
        )
        assert r.objective == pytest.approx(3.0, abs=1e-5)

    def test_general_integer_optimum_at_larger_value(self):
        """A general-integer optimum away from 0/1: y*z >= 4, z in [0, 5] → (1, 4)."""
        m = dm.Model("loa_nogood_int_large")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=5)
        m.subject_to(y * z >= 4)
        m.minimize(y + z)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(5.0, abs=1e-5)
        assert float(np.asarray(r.x["z"])) == pytest.approx(4.0, abs=1e-4)

    def test_two_general_integers(self):
        """Two general integers: a*b >= 6, a, b in [0, 4] → optimum 5."""
        m = dm.Model("loa_two_int")
        a = m.integer("a", lb=0, ub=4)
        b = m.integer("b", lb=0, ub=4)
        m.subject_to(a * b >= 6)
        m.minimize(a + b)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(5.0, abs=1e-5)

    def test_genuinely_infeasible_general_integer_still_infeasible(self):
        """The fix must not over-correct: a truly infeasible integer model stays
        ``"infeasible"`` (max y*z = 3 < 100)."""
        m = dm.Model("loa_int_infeasible")
        y = m.binary("y")
        z = m.integer("z", lb=0, ub=3)
        m.subject_to(y * z >= 100)
        m.minimize(y + z)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status == "infeasible"
        assert r.objective is None


@pytest.mark.unit
class TestIssue756Augmentation:
    def test_discrete_vars_all_binary(self):
        lb = np.array([0.0, 0.0])
        assert loa._discrete_vars_all_binary([0, 1], lb, np.array([1.0, 1.0]))
        assert not loa._discrete_vars_all_binary([0, 1], lb, np.array([1.0, 3.0]))

    def test_exact_exclusion_removes_only_target_point(self):
        """The exact no-good excludes EXACTLY v* and keeps every other integer point.

        Enumerate the box y in {0,1}, z in {0..3} against the augmented system
        (variables + auxiliary binaries) after excluding (0, 1).
        """
        int_indices = [0, 1]
        lb = np.array([0.0, 0.0])
        ub = np.array([1.0, 3.0])
        rows, rhs, n_aux = loa._build_no_good_augmentation([(0, 1)], int_indices, lb, ub, n_base=2)
        A = np.array(rows)
        b = np.array(rhs)

        surviving = set()
        for yv, zv in itertools.product([0, 1], [0, 1, 2, 3]):
            for aux in itertools.product([0, 1], repeat=n_aux):
                full = np.array([yv, zv, *aux], dtype=float)
                if np.all(A @ full <= b + 1e-9):
                    surviving.add((yv, zv))
                    break

        all_points = set(itertools.product([0, 1], [0, 1, 2, 3]))
        assert (0, 1) not in surviving, "target point (0,1) must be excluded"
        assert surviving == all_points - {(0, 1)}, "no other point may be cut"

    def test_fully_pinned_config_makes_master_infeasible(self):
        """A config pinned at its bounds in every coordinate yields the empty
        aggregate row ``0 <= -1`` — correct, since v* was the only integer point."""
        rows, rhs, n_aux = loa._build_no_good_augmentation(
            [(2,)], [0], np.array([2.0]), np.array([2.0]), n_base=1
        )
        assert n_aux == 0
        # one aggregate row, all-zero LHS with rhs -1  ->  0 <= -1 (infeasible)
        assert len(rows) == 1
        np.testing.assert_allclose(rows[0], [0.0])
        assert rhs[0] == -1.0


@pytest.mark.smoke
class TestIssue756TimeLimit:
    def test_time_limit_exhausted_is_not_infeasible(self):
        """#756(2): a zero-iteration timeout must not certify infeasibility."""
        m = dm.Model("loa_tl0")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.minimize(x)
        r = loa.solve_gdpopt_loa(m, time_limit=0.0)
        assert r.status != "infeasible", (
            "timeout with zero iterations must not certify infeasibility"
        )
        assert r.status == "unknown"
        assert r.objective is None
        assert r.gap_certified is False


@pytest.mark.smoke
class TestIssue756NoBinaryRegression:
    """The all-binary path is unchanged: these must keep their prior behavior."""

    def test_binary_no_good_cut_reaches_optimum(self):
        m = dm.Model("loa_nogood_bin")
        y1 = m.binary("y1")
        y2 = m.binary("y2")
        m.subject_to(y1 * y2 >= 1)
        m.minimize(y1 + y2)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status in ("optimal", "feasible")
        assert r.objective == pytest.approx(2.0, abs=1e-5)

    def test_infeasible_by_master_still_infeasible(self):
        m = dm.Model("loa_infeas")
        x = m.continuous("x", lb=0, ub=10)
        m.either_or([[x <= 3], [x >= 7]], name="choice")
        m.subject_to(x >= 4)
        m.subject_to(x <= 6)
        m.minimize(x)
        r = m.solve(time_limit=30, gdp_method="loa")
        assert r.status == "infeasible"
        assert r.objective is None

    def test_add_no_good_cut_binary_form_unchanged(self):
        A_rows: list = []
        b_rows: list = []
        loa._add_no_good_cut(np.array([1.0, 0.0]), [0, 1], A_rows, b_rows, 2)
        np.testing.assert_allclose(A_rows[0], [1.0, -1.0])
        assert b_rows[0] == 0.0
