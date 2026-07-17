"""Tests for the GP-structured MINLP path (issue #116).

A MINLP whose *continuous relaxation* is a geometric program is solved by
integer branch-and-bound in which every node relaxation is the exact convex
log-space NLP (``yᵢ = log xᵢ`` per node, bounds mapped
``xᵢ ∈ [lᵢ, uᵢ] → yᵢ ∈ [log lᵢ, log uᵢ]``). Each node bound is a rigorous
convex-GP bound, so a closed tree certifies a global optimum.

Covers: classification (positive-lower-bound integers admitted, binaries /
unbounded integers / non-GP relaxations refused), end-to-end solves against
known optima, a differential panel against the independent classic spatial
branch-and-bound, certified infeasibility, and the ``solver="gp-minlp"``
dispatch + ``DISCOPT_GP_MINLP`` auto-route.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.gp import (
    classify_gp,
    classify_gp_minlp,
    is_gp_minlp,
    is_log_convex,
    solve_gp_minlp,
)
from discopt.modeling.core import Model


def _val(x: dict, name: str) -> float:
    return float(np.asarray(x[name]))


# ──────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────


class TestClassification:
    def test_integer_gp_is_gp_minlp(self):
        m = Model("gpint")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        info = classify_gp_minlp(m)
        assert info is not None
        assert is_gp_minlp(m)
        # The single scalar integer occupies flat offset 0.
        assert info.integer_offsets == [0]
        assert info.structure.minimize

    def test_mixed_continuous_integer_is_gp_minlp(self):
        m = Model("mix")
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x + n / x + n)
        info = classify_gp_minlp(m)
        assert info is not None
        # x is offset 0 (continuous), n is offset 1 (integer).
        assert info.integer_offsets == [1]

    def test_pure_continuous_gp_is_not_gp_minlp(self):
        # No integer variable -> handled by the pure-GP path, not this one.
        m = Model("puregp")
        x = m.continuous("x", lb=0.1, ub=10.0)
        m.minimize(x + 1.0 / x)
        assert classify_gp_minlp(m) is None
        assert not is_gp_minlp(m)
        # ... but it is still a plain GP.
        assert classify_gp(m) is not None

    def test_binary_variable_is_not_gp_minlp(self):
        # A binary {0,1} has lb == 0, so the log map yᵢ = log xᵢ is undefined:
        # refuse (fall back to spatial B&B) rather than reason unsoundly.
        m = Model("bin")
        x = m.continuous("x", lb=0.1, ub=10.0)
        b = m.binary("b")
        m.minimize(x + b / x + b)
        assert classify_gp_minlp(m) is None

    def test_zero_lb_integer_is_not_gp_minlp(self):
        m = Model("intzero")
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=0, ub=4)  # lb == 0 breaks the log map
        m.minimize(x + n / x + n)
        assert classify_gp_minlp(m) is None

    def test_unbounded_integer_is_not_gp_minlp(self):
        # An integer variable with no finite upper bound would make B&B
        # non-terminating: refuse.
        m = Model("intinf")
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=np.inf)
        m.minimize(x + n / x)
        assert classify_gp_minlp(m) is None

    def test_signomial_relaxation_is_not_gp_minlp(self):
        # Continuous relaxation is a signomial (negative monomial) -> not GP.
        m = Model("sig")
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x * n)
        m.subject_to(x * n - 3.0 * x <= 1.0)
        assert classify_gp_minlp(m) is None

    def test_gp_minlp_keeps_is_log_convex_false(self):
        # is_log_convex is deliberately whole-model *continuous* GP recognition;
        # an integer model must still report False there (issue #116 preamble).
        m = Model("gpint")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        assert is_log_convex(m) is False


# ──────────────────────────────────────────────────────────────────────
# End-to-end solves against known optima
# ──────────────────────────────────────────────────────────────────────


class TestSolve:
    def test_pure_integer_relaxation_is_fractional(self):
        # minimize n + 2.25/n, n integer in [1, 5].
        # Continuous relaxation optimum is n = 1.5 (fractional), value 3.0;
        # the integer optimum is n = 2, value 2 + 2.25/2 = 3.125. The solve
        # must branch off the fractional root and return the integer optimum.
        m = Model("gpint")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = solve_gp_minlp(m)
        assert r is not None
        assert r.status == "optimal"
        assert r.objective == pytest.approx(3.125, abs=1e-5)
        assert _val(r.x, "n") == pytest.approx(2.0, abs=1e-6)
        # Branched: root fractional -> two children -> 3 nodes.
        assert r.node_count >= 2
        # Certified global optimum: zero gap, bound == incumbent.
        assert r.gap_certified is True
        assert r.gap == pytest.approx(0.0, abs=1e-9)
        assert r.bound == pytest.approx(r.objective, abs=1e-6)
        # Certificate invariant: bound <= incumbent for a minimisation.
        assert r.bound <= r.objective + 1e-9

    def test_integral_root_solves_in_one_node(self):
        # minimize x + n/x + n over x in [0.1,10], n integer [1,4].
        # For fixed n the min over x is at x = sqrt(n); the whole thing is
        # increasing in n, so the optimum is n = 1, x = 1, value 3 — and the
        # relaxation already returns an integral n, so no branching is needed.
        m = Model("mix")
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x + n / x + n)
        r = solve_gp_minlp(m)
        assert r is not None
        assert r.status == "optimal"
        assert r.objective == pytest.approx(3.0, abs=1e-4)
        assert _val(r.x, "n") == pytest.approx(1.0, abs=1e-6)
        assert r.node_count == 1
        assert r.gap_certified is True

    def test_monomial_maximisation(self):
        # maximize x*n s.t. x*n <= 6.5, x in [0.1,10], n integer [1,4].
        # The bound is tight: optimum value 6.5 (e.g. n=1, x=6.5).
        m = Model("maxgp")
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.maximize(x * n)
        m.subject_to(x * n <= 6.5)
        r = solve_gp_minlp(m)
        assert r is not None
        assert r.status == "optimal"
        assert r.objective == pytest.approx(6.5, abs=1e-4)
        assert r.gap_certified is True
        # Certificate invariant: bound >= incumbent for a maximisation.
        assert r.bound >= r.objective - 1e-6

    def test_certified_infeasible(self):
        # x*n >= 100 is impossible with x <= 10, n <= 4 (max product 40).
        # Every node relaxation is infeasible -> the MINLP is certifiably
        # infeasible.
        m = Model("infeas")
        x = m.continuous("x", lb=1.0, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x)
        m.subject_to(100.0 / (x * n) <= 1.0)
        r = solve_gp_minlp(m)
        assert r is not None
        assert r.status == "infeasible"
        assert r.objective is None
        assert r.gap_certified is True

    def test_returns_none_for_non_gp_minlp(self):
        # A pure continuous GP is not this path's job.
        m = Model("puregp")
        x = m.continuous("x", lb=0.1, ub=10.0)
        m.minimize(x + 1.0 / x)
        assert solve_gp_minlp(m) is None

    def test_gap_tolerance_early_fathom_stays_valid(self):
        # minimize n + 2.25/n, n in [1,5]. The root relaxation bound is 3.0; the
        # first incumbent (n=1) is 3.25, a 7.7% gap. A loose 10% tolerance lets
        # the search fathom there — the result must remain sound: a valid dual
        # bound (<= true optimum), a certified gap within the tolerance, and a
        # genuinely feasible incumbent.
        m = Model("looose")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = solve_gp_minlp(m, gap_tolerance=0.1)
        assert r is not None
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert r.gap is not None and r.gap <= 0.1 + 1e-9
        assert r.bound is not None
        assert r.bound <= 3.125 + 1e-9  # valid lower bound on the true optimum
        assert r.bound <= r.objective + 1e-9  # certificate invariant

    def test_tight_gap_reaches_true_optimum(self):
        # The default tight tolerance drives all the way to the true optimum.
        m = Model("tight")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = solve_gp_minlp(m, gap_tolerance=1e-6)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(3.125, abs=1e-5)
        assert r.gap == pytest.approx(0.0, abs=1e-6)

    def test_non_convergent_node_stays_sound(self, monkeypatch):
        # If a node relaxation fails to converge, its subtree is abandoned. The
        # result must NOT be certified, and the reported dual bound must remain a
        # *valid* lower bound (<= true optimum), never the incumbent masquerading
        # as one — even though the abandoned subtree here holds the real optimum.
        import discopt.gp as gp

        # minimize n + 2.25/n, n in [1,5]: root n=1.5 -> children [1,1] and [2,5].
        # Force the up-child ([2,5], which contains the optimum n=2, obj 3.125)
        # to "fail", so the search only ever sees n=1 (obj 3.25).
        real = gp._solve_gp_node

        def flaky(structure, x_lb, x_ub, node_index, solve_kwargs):
            if float(x_lb[0]) >= 2.0:  # the n >= 2 branch
                return "iteration_limit", None, None
            return real(structure, x_lb, x_ub, node_index, solve_kwargs)

        monkeypatch.setattr(gp, "_solve_gp_node", flaky)

        m = Model("flaky")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = solve_gp_minlp(m)
        assert r is not None
        assert r.gap_certified is False  # abandoned subtree -> not certified
        assert r.objective == pytest.approx(3.25, abs=1e-4)  # the n=1 incumbent
        # The dual bound is still valid: <= the true optimum (3.125), which the
        # abandoned subtree actually contained.
        assert r.bound is not None
        assert r.bound <= 3.125 + 1e-6


# ──────────────────────────────────────────────────────────────────────
# Differential panel: agree with the independent classic spatial B&B
# ──────────────────────────────────────────────────────────────────────


def _panel(kind: str) -> Model:
    m = Model(kind)
    if kind == "a":
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
    elif kind == "b":
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x + n / x + n)
    elif kind == "c":
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x * n + 9.0 / (x * n))
    elif kind == "d":
        x = m.continuous("x", lb=0.1, ub=10.0)
        n = m.integer("n", lb=1, ub=6)
        m.minimize(x + 4.0 / x + 3.0 * n + 12.0 / n)
    elif kind == "e":
        a = m.integer("a", lb=1, ub=5)
        b = m.integer("b", lb=1, ub=5)
        m.minimize(a * b + 6.0 / a + 6.0 / b)
    elif kind == "f":
        x = m.continuous("x", lb=0.5, ub=8.0)
        n = m.integer("n", lb=1, ub=4)
        m.minimize(x + 5.0 / (x * n))
        m.subject_to(x * n <= 6.0)
    else:  # pragma: no cover
        raise ValueError(kind)
    return m


class TestDifferentialAgainstClassicBB:
    @pytest.mark.parametrize("kind", ["a", "b", "c", "d", "e", "f"])
    def test_gp_minlp_matches_spatial_bb(self, kind):
        gp = solve_gp_minlp(_panel(kind))
        bb = _panel(kind).solve(solver="bb")
        assert gp is not None and gp.objective is not None
        assert bb.objective is not None
        # The two independent solvers must reach the same certified optimum.
        assert gp.objective == pytest.approx(bb.objective, rel=1e-4, abs=1e-4)
        assert gp.gap_certified is True
        # bound is a valid lower bound on the incumbent (min sense).
        assert gp.bound is not None
        assert gp.bound <= gp.objective + 1e-6


# ──────────────────────────────────────────────────────────────────────
# Solver dispatch: model.solve(solver="gp-minlp") and the auto-route flag
# ──────────────────────────────────────────────────────────────────────


class TestSolverDispatch:
    def test_explicit_solver_matches_direct(self):
        m = Model("gpint")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = m.solve(solver="gp-minlp")
        assert r.status == "optimal"
        assert r.objective == pytest.approx(3.125, abs=1e-5)
        assert r.gap_certified is True

    def test_explicit_solver_rejects_non_gp_minlp(self):
        m = Model("notgp")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        m.minimize(x * x)
        with pytest.raises(ValueError, match="not a .*GP-structured MINLP"):
            m.solve(solver="gp-minlp")

    def test_auto_route_off_by_default(self, monkeypatch):
        # With the flag unset, a plain solve() must NOT take the GP-MINLP path;
        # the classic B&B still solves it correctly (the two must agree).
        monkeypatch.delenv("DISCOPT_GP_MINLP", raising=False)
        m = Model("gpint")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = m.solve()
        assert r.objective == pytest.approx(3.125, abs=1e-4)

    def test_auto_route_on_with_flag(self, monkeypatch):
        monkeypatch.setenv("DISCOPT_GP_MINLP", "1")
        m = Model("gpint")
        n = m.integer("n", lb=1, ub=5)
        m.minimize(n + 2.25 / n)
        r = m.solve()
        assert r.status == "optimal"
        assert r.objective == pytest.approx(3.125, abs=1e-5)
        assert r.gap_certified is True
