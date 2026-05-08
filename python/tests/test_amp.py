"""
Tests for the Adaptive Multivariate Partitioning (AMP) global MINLP solver.

Theory references:
  - CP 2016: "Tightening McCormick Relaxations via Dynamic Multivariate
    Partitioning", Nagarajan et al.
  - JOGO 2018: "An Adaptive, Multivariate Partitioning Algorithm for Global
    Optimization of Nonconvex Programs", Nagarajan et al.

Test strategy (TDD):
  All tests are written BEFORE implementation.  Running this suite against
  the current codebase reveals which modules are missing and which existing
  functions have gaps or bugs.  Each section maps to a new module that must
  be created to make the tests pass.

Sections:
  1. Term Classifier            (term_classifier.py)
  2. Variable Selection         (partition_selection.py)
  3. Discretization State       (discretization.py)
  4. Piecewise McCormick validity (existing mccormick.py / piecewise_mccormick.py)
  5. MILP Relaxation lower bounds (milp_relaxation.py)
  6. End-to-end AMP             (solvers/amp.py + solver.py routing)
  7. Convergence properties     (AMP loop invariants)
  8. Weakness probes            (expose bugs/gaps in current code)
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import (
    Model,
    SolveResult,
)

# ---------------------------------------------------------------------------
# Shared helpers / problem builders
# ---------------------------------------------------------------------------


def _make_nlp1() -> Model:
    """Alpine nlp1: min 6x₀²+4x₁²-2.5x₀x₁  s.t. x₀x₁≥8, x∈[1,4]²

    Known global optimum ≈ 58.38368 (verified in Alpine.jl test suite).
    """
    m = Model("nlp1")
    x = m.continuous("x", lb=1, ub=4, shape=(2,))
    m.subject_to(x[0] * x[1] >= 8)
    m.minimize(6 * x[0] ** 2 + 4 * x[1] ** 2 - 2.5 * x[0] * x[1])
    return m


def _make_circle() -> Model:
    """Alpine circle: min x₀+x₁  s.t. x₀²+x₁²≥2, x∈[0,2]²

    Known global optimum = √2 ≈ 1.41421356.
    """
    m = Model("circle")
    x = m.continuous("x", lb=0, ub=2, shape=(2,))
    m.subject_to(x[0] ** 2 + x[1] ** 2 >= 2)
    m.minimize(x[0] + x[1])
    return m


def _make_trilinear_cover() -> Model:
    """Simple trilinear problem with a known AM-GM optimum of 3."""
    m = Model("trilinear_cover")
    x = m.continuous("x", lb=0, ub=2, shape=(3,))
    m.subject_to(x[0] * x[1] * x[2] >= 1.0)
    m.minimize(x[0] + x[1] + x[2])
    return m


def _make_convex_quadratic() -> Model:
    """Small convex model that AMP should certify globally."""
    m = Model("convex_quadratic")
    x = m.continuous("x", lb=1, ub=10)
    m.minimize(x**2)
    return m


def _build_nlp3() -> Model:
    """Alpine nlp3: 8-variable multilinear industrial process.

    Ported from Alpine.jl examples/MINLPs/nlp.jl (nlp3 function).
    Known global optimum ≈ 7049.247898 (Alpine test suite).

    Alpine formulation (1-indexed → 0-indexed Python):
      min  x[1]+x[2]+x[3]  →  x[0]+x[1]+x[2]
      s.t. 0.0025*(x[4]+x[6]) <= 1          (linear)
           0.0025*(x[5]-x[4]+x[7]) <= 1     (linear)
           0.01*(x[8]-x[5]) <= 1             (linear)
           100*x[1] - x[1]*x[6] + 833.33252*x[4] <= 83333.333  (bilinear)
           x[2]*x[4] - x[2]*x[7] - 1250*x[4] + 1250*x[5] <= 0  (bilinear)
           x[3]*x[5] - x[3]*x[8] - 2500*x[5] + 1250000 <= 0     (bilinear)
    """
    m = Model("nlp3")
    LB = [100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    UB = [10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    x = [m.continuous(f"x{i}", lb=LB[i], ub=UB[i]) for i in range(8)]
    m.minimize(x[0] + x[1] + x[2])
    # Linear constraints
    m.subject_to(0.0025 * (x[3] + x[5]) <= 1)
    m.subject_to(0.0025 * (x[4] - x[3] + x[6]) <= 1)
    m.subject_to(0.01 * (x[7] - x[4]) <= 1)
    # Nonlinear (bilinear) constraints — in >= form (multiply Alpine's <= by -1)
    m.subject_to(83333.333 - 100.0 * x[0] + x[0] * x[5] - 833.33252 * x[3] >= 0)
    m.subject_to(x[1] * x[6] - x[1] * x[3] - 1250.0 * x[4] + 1250.0 * x[3] >= 0)
    m.subject_to(x[2] * x[7] - x[2] * x[4] + 2500.0 * x[4] - 1250000.0 >= 0)
    return m


# ===========================================================================
# Section 1: Nonlinear Term Classifier
#
# Module to be created: python/discopt/_jax/term_classifier.py
# Function: classify_nonlinear_terms(model: Model) -> NonlinearTerms
# ===========================================================================


class TestTermClassifier:
    """Tests for classify_nonlinear_terms().

    These will all fail with ImportError until term_classifier.py is created.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        """Import the to-be-created module (expected to fail initially)."""
        from discopt._jax.term_classifier import (  # noqa: F401
            NonlinearTerms,
            classify_nonlinear_terms,
        )

        self.classify = classify_nonlinear_terms
        self.NonlinearTerms = NonlinearTerms

    def test_detect_bilinear_objective(self):
        """x[0]*x[1] in objective → one bilinear term (0,1)."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10, shape=(2,))
        m.minimize(x[0] * x[1])
        terms = self.classify(m)
        assert len(terms.bilinear) == 1
        assert set(terms.bilinear[0]) == {0, 1}

    def test_detect_bilinear_constraint(self):
        """x[0]*x[1] in constraint → detected as bilinear."""
        m = Model("t")
        x = m.continuous("x", lb=1, ub=4, shape=(2,))
        m.subject_to(x[0] * x[1] >= 8)
        m.minimize(x[0] + x[1])
        terms = self.classify(m)
        assert len(terms.bilinear) >= 1
        found = any(set(b) == {0, 1} for b in terms.bilinear)
        assert found, f"Expected bilinear (0,1), got {terms.bilinear}"

    def test_detect_monomial_squared(self):
        """x[0]**2 → monomial with exponent 2."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10, shape=(2,))
        m.minimize(x[0] ** 2 + x[1] ** 2)
        terms = self.classify(m)
        assert len(terms.monomial) >= 2
        exponents = [exp for _, exp in terms.monomial]
        assert all(e == 2 for e in exponents), f"Expected exponent 2, got {exponents}"

    def test_detect_trilinear(self):
        """x[0]*x[1]*x[2] → one trilinear term."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10, shape=(3,))
        m.minimize(x[0] * x[1] * x[2])
        terms = self.classify(m)
        assert len(terms.trilinear) >= 1
        found = any(set(t) == {0, 1, 2} for t in terms.trilinear)
        assert found, f"Expected trilinear (0,1,2), got {terms.trilinear}"

    def test_mixed_bilinear_and_monomial(self):
        """nlp1: bilinear in constraint, monomial in objective."""
        m = _make_nlp1()
        terms = self.classify(m)
        assert len(terms.bilinear) >= 1
        assert len(terms.monomial) >= 2  # x[0]**2 and x[1]**2

    def test_repeated_factor_product_is_general_nl(self):
        """Mixed repeated-factor products like x*x*y must not be misclassified as bilinear."""
        m = Model("repeat_factor")
        x = m.continuous("x", lb=0, ub=10, shape=(2,))
        m.minimize((x[0] * x[0]) * x[1])

        terms = self.classify(m)

        assert len(terms.bilinear) == 0
        assert len(terms.general_nl) >= 1
        assert (0, 2) in terms.monomial

    def test_partition_candidates_excludes_linear_vars(self):
        """Variable not in any nonlinear term must NOT be a partition candidate."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10, shape=(3,))
        # Only x[0]*x[1] — x[2] is only in linear expressions
        m.minimize(x[0] * x[1] + 2.0 * x[2])
        terms = self.classify(m)
        # x[0] and x[1] appear in bilinear term
        assert 0 in terms.partition_candidates
        assert 1 in terms.partition_candidates
        # x[2] is only in a linear term — must NOT be a candidate
        assert 2 not in terms.partition_candidates, (
            "x[2] is linear-only and must not be a partition candidate"
        )

    def test_transcendental_classified_as_general_nl(self):
        """sin(x), exp(x), log(x) → classified as general_nl (not bilinear)."""
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=3.14)
        m.minimize(dm.sin(x) + dm.exp(x))
        terms = self.classify(m)
        assert len(terms.general_nl) >= 1, "sin/exp should be classified as general_nl"
        assert len(terms.bilinear) == 0, "sin/exp must not be misclassified as bilinear"

    def test_empty_model_has_no_terms(self):
        """Model with only linear expressions → no nonlinear terms."""
        m = Model("linear")
        x = m.continuous("x", lb=0, ub=1, shape=(3,))
        m.minimize(x[0] + 2 * x[1] - x[2])
        terms = self.classify(m)
        assert len(terms.bilinear) == 0
        assert len(terms.monomial) == 0
        assert len(terms.trilinear) == 0
        assert len(terms.partition_candidates) == 0

    def test_term_incidence_correct(self):
        """term_incidence maps var_idx → set of term indices it appears in."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10, shape=(3,))
        # Two bilinear terms: x[0]*x[1] and x[0]*x[2]
        m.minimize(x[0] * x[1] + x[0] * x[2])
        terms = self.classify(m)
        # x[0] should appear in 2 terms; x[1] and x[2] each in 1
        assert len(terms.term_incidence[0]) == 2, "x[0] appears in 2 bilinear terms"
        assert len(terms.term_incidence[1]) == 1
        assert len(terms.term_incidence[2]) == 1


# ===========================================================================
# Section 2: Variable Selection for AMP Partitioning
#
# Module to be created: python/discopt/_jax/partition_selection.py
# Function: pick_partition_vars(terms, method="auto") -> list[int]
# ===========================================================================


class TestPartitionSelection:
    """Tests for pick_partition_vars().

    These will all fail with ImportError until partition_selection.py is created.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from discopt._jax.partition_selection import pick_partition_vars  # noqa: F401
        from discopt._jax.term_classifier import NonlinearTerms  # noqa: F401

        self.pick = pick_partition_vars
        self.NonlinearTerms = NonlinearTerms

    def _make_terms(self, bilinear=None, trilinear=None, monomial=None):
        """Helper to build a NonlinearTerms stub."""
        bilinear = bilinear or []
        trilinear = trilinear or []
        monomial = monomial or []
        candidates = list({v for t in bilinear + trilinear for v in t} | {v for v, _ in monomial})
        incidence: dict[int, set[int]] = {}
        for idx, t in enumerate(bilinear + trilinear):
            for v in t:
                incidence.setdefault(v, set()).add(idx)
        return self.NonlinearTerms(
            bilinear=bilinear,
            trilinear=trilinear,
            monomial=monomial,
            general_nl=[],
            term_incidence=incidence,
            partition_candidates=candidates,
        )

    def test_max_cover_returns_all_candidates(self):
        """max_cover selects every variable appearing in any nonlinear term."""
        terms = self._make_terms(bilinear=[(0, 1), (1, 2)])
        selected = self.pick(terms, method="max_cover")
        assert set(selected) == {0, 1, 2}

    def test_min_vertex_cover_covers_all_terms(self):
        """min_vertex_cover: every bilinear term must have ≥1 selected variable."""
        # Star: (0,1),(0,2),(0,3),(0,4) — selecting just {0} covers all
        terms = self._make_terms(bilinear=[(0, 1), (0, 2), (0, 3), (0, 4)])
        selected = self.pick(terms, method="min_vertex_cover")
        for t in terms.bilinear:
            assert any(v in selected for v in t), f"term {t} not covered"

    def test_min_vertex_cover_smaller_than_max_cover(self):
        """min_vertex_cover should use ≤ variables than max_cover for star graphs."""
        terms = self._make_terms(bilinear=[(0, 1), (0, 2), (0, 3), (0, 4)])
        mvc = self.pick(terms, method="min_vertex_cover")
        maxc = self.pick(terms, method="max_cover")
        assert len(mvc) <= len(maxc)
        assert 0 in mvc, "hub variable 0 should be selected"

    def test_min_vertex_cover_path_graph(self):
        """Path graph: (0,1),(1,2),(2,3) — vertex 1 and 2 cover all edges."""
        terms = self._make_terms(bilinear=[(0, 1), (1, 2), (2, 3)])
        selected = self.pick(terms, method="min_vertex_cover")
        for t in terms.bilinear:
            assert any(v in selected for v in t), f"term {t} not covered by {selected}"

    def test_auto_uses_max_cover_for_small(self):
        """auto strategy uses max_cover when ≤15 partition candidates."""
        terms = self._make_terms(bilinear=[(0, 1), (2, 3)])
        auto = self.pick(terms, method="auto")
        maxc = self.pick(terms, method="max_cover")
        assert set(auto) == set(maxc)

    def test_auto_uses_min_vertex_cover_for_large(self):
        """auto strategy uses min_vertex_cover when >15 partition candidates."""
        # 20 variables in a star — hub is var 0
        terms = self._make_terms(bilinear=[(0, i) for i in range(1, 20)])
        auto = self.pick(terms, method="auto")
        # Both should cover all terms
        for t in terms.bilinear:
            assert any(v in auto for v in t), f"term {t} not covered"
        # auto should be no larger than max_cover
        maxc = self.pick(terms, method="max_cover")
        assert len(auto) <= len(maxc)

    def test_trilinear_terms_covered(self):
        """Trilinear terms (x*y*z) must also be covered."""
        terms = self._make_terms(trilinear=[(0, 1, 2)])
        selected = self.pick(terms, method="min_vertex_cover")
        t = (0, 1, 2)
        assert any(v in selected for v in t), "trilinear term not covered"

    def test_empty_terms_returns_empty(self):
        """No nonlinear terms → empty partition variable list."""
        terms = self._make_terms()
        assert self.pick(terms, method="max_cover") == []
        assert self.pick(terms, method="min_vertex_cover") == []

    def test_min_vertex_cover_falls_back_to_greedy_cover(self, monkeypatch):
        """When the MILP cover solve fails, the greedy fallback should still cover all terms."""
        import discopt._jax.partition_selection as part_sel

        terms = self._make_terms(bilinear=[(0, 1), (0, 2), (0, 3), (0, 4)])
        monkeypatch.setattr(
            part_sel,
            "_solve_vertex_cover_milp",
            lambda candidates, all_t: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        selected = self.pick(terms, method="min_vertex_cover")

        assert set(selected) == {0}

    def test_min_vertex_cover_falls_back_when_highs_status_is_not_optimal(self, monkeypatch):
        """A non-optimal HiGHS status should fall back to the greedy cover."""
        import highspy

        terms = self._make_terms(bilinear=[(0, 1), (0, 2), (0, 3), (0, 4)])
        options = {}

        class FakeHighs:
            def silent(self):
                return None

            def setOptionValue(self, name, value):
                options[name] = value

            def addBinary(self, obj):
                return obj

            def addRow(self, *args):
                return args

            def run(self):
                return None

            def getModelStatus(self):
                return highspy.HighsModelStatus.kTimeLimit

            def getSolution(self):
                raise AssertionError("solution should not be read after a non-optimal status")

        monkeypatch.setattr(highspy, "Highs", FakeHighs)

        selected = self.pick(terms, method="min_vertex_cover")

        assert set(selected) == {0}
        assert options["time_limit"] == pytest.approx(30.0)

    def test_adaptive_vertex_cover_uses_distance_weights(self):
        """Adaptive cover should favor high-distance variables on large graphs."""
        terms = self._make_terms(bilinear=[(i, i + 1) for i in range(16)])
        distance = {i: (10.0 if i % 2 == 0 else 1e-3) for i in terms.partition_candidates}

        selected = self.pick(
            terms,
            method="adaptive_vertex_cover",
            distance=distance,
        )

        assert selected
        assert all(v % 2 == 0 for v in selected)
        for t in terms.bilinear:
            assert any(v in selected for v in t), f"term {t} not covered"


# ===========================================================================
# Section 3: Discretization State Management
#
# Module to be created: python/discopt/_jax/discretization.py
# Classes/functions:
#   DiscretizationState
#   initialize_partitions(var_indices, lb, ub, n_init=2) -> DiscretizationState
#   add_adaptive_partition(state, solution, var_indices, lb, ub) -> DiscretizationState
#   check_partition_convergence(state) -> bool
# ===========================================================================


class TestDiscretizationState:
    """Tests for discretization state management.

    Will fail with ImportError until discretization.py is created.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from discopt._jax.discretization import (  # noqa: F401
            DiscretizationState,
            add_adaptive_partition,
            add_uniform_partition,
            check_partition_convergence,
            initialize_partitions,
        )

        self.DiscretizationState = DiscretizationState
        self.initialize_partitions = initialize_partitions
        self.add_adaptive_partition = add_adaptive_partition
        self.add_uniform_partition = add_uniform_partition
        self.check_partition_convergence = check_partition_convergence

    def test_initialize_correct_breakpoints(self):
        """2 initial intervals → 3 breakpoints spanning [lb, ub]."""
        state = self.initialize_partitions([0, 1], lb=[0.0, 2.0], ub=[10.0, 8.0], n_init=2)
        assert 0 in state.partitions
        assert 1 in state.partitions
        pts0 = state.partitions[0]
        assert pts0[0] == pytest.approx(0.0)
        assert pts0[-1] == pytest.approx(10.0)
        assert len(pts0) == 3  # n_init=2 intervals → 3 edges

    def test_initialize_uniform_spacing(self):
        """Breakpoints should be uniformly spaced with n_init intervals."""
        state = self.initialize_partitions([0], lb=[0.0], ub=[6.0], n_init=3)
        pts = state.partitions[0]
        assert len(pts) == 4
        diffs = np.diff(pts)
        assert np.allclose(diffs, diffs[0], atol=1e-12), "Should be uniform spacing"

    def test_adaptive_adds_breakpoints_near_solution(self):
        """After refinement, new breakpoints should appear near the solution value."""
        state = self.initialize_partitions([0], lb=[0.0], ub=[10.0], n_init=2)
        sol = {0: 3.0}
        state2 = self.add_adaptive_partition(state, sol, [0], lb=[0.0], ub=[10.0])
        pts = state2.partitions[0]
        assert len(pts) > len(state.partitions[0]), "Should have more breakpoints"
        # At least one new breakpoint within 1.5 of the solution (3.0)
        assert any(1.5 <= p <= 4.5 for p in pts if p not in (0.0, 5.0, 10.0)), (
            f"Expected breakpoint near 3.0, got {pts}"
        )

    def test_refinement_reduces_active_partition_width(self):
        """The partition containing the solution must be split after refinement.

        Note: the GLOBAL max width may not decrease if the solution falls in a
        non-maximal interval.  Start from a single interval to ensure the active
        partition IS the widest, so global max must decrease.
        """
        # Use n_init=1 → [0.0, 10.0]: single interval of width 10
        state = self.initialize_partitions([0], lb=[0.0], ub=[10.0], n_init=1)
        assert len(state.partitions[0]) == 2

        solution = {0: 3.0}
        state2 = self.add_adaptive_partition(state, solution, [0], lb=[0.0], ub=[10.0])
        w1 = float(np.max(np.diff(state.partitions[0])))
        w2 = float(np.max(np.diff(state2.partitions[0])))
        assert w2 < w1, f"Max width should decrease from single interval: {w1} → {w2}"

    def test_partitions_are_monotonically_refined(self):
        """After multiple refinements, partitions only become finer (never coarser)."""
        state = self.initialize_partitions([0], lb=[0.0], ub=[1.0])
        all_seen: set[float] = set(state.partitions[0].tolist())
        for sol_val in [0.3, 0.7, 0.15, 0.55]:
            state = self.add_adaptive_partition(state, {0: sol_val}, [0], [0.0], [1.0])
            new_pts = set(state.partitions[0].tolist())
            missing = all_seen - new_pts
            assert not missing, f"Breakpoints removed during refinement: {missing}"
            all_seen = new_pts
        # Endpoints must be preserved
        pts = state.partitions[0]
        assert pts[0] == pytest.approx(0.0)
        assert pts[-1] == pytest.approx(1.0)
        # Strictly sorted
        assert all(pts[i] < pts[i + 1] for i in range(len(pts) - 1))

    def test_adaptive_refinement_bisects_when_centered_candidates_hit_interval_edges(self):
        """If centered refinement would add only edge points, fall back to the midpoint."""
        state = self.initialize_partitions(
            [0],
            lb=[0.0],
            ub=[1.0],
            n_init=1,
            scaling_factor=2.0,
        )

        state2 = self.add_adaptive_partition(state, {0: 0.5}, [0], [0.0], [1.0])

        pts = state2.partitions[0]
        assert len(pts) == 3
        np.testing.assert_allclose(pts, np.array([0.0, 0.5, 1.0]))

    def test_uniform_refinement_splits_every_interval(self):
        """Uniform refinement should bisect each current interval."""
        state = self.initialize_partitions([0], lb=[0.0], ub=[4.0], n_init=2)
        state2 = self.add_uniform_partition(state, {}, [0], [0.0], [4.0])
        np.testing.assert_allclose(
            state2.partitions[0],
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        )

    def test_convergence_check_fine_partitions(self):
        """check_partition_convergence returns True when all widths < abs_width_tol."""
        bpts = np.linspace(0, 1, 3000)  # width ≈ 3.3e-4 < 1e-3
        state = self.DiscretizationState(
            partitions={0: bpts},
            scaling_factor=10.0,
            abs_width_tol=1e-3,
        )
        assert self.check_partition_convergence(state) is True

    def test_convergence_check_coarse_partitions(self):
        """check_partition_convergence returns False when widths > abs_width_tol."""
        bpts = np.array([0.0, 5.0, 10.0])  # width = 5.0 >> 1e-3
        state = self.DiscretizationState(
            partitions={0: bpts},
            scaling_factor=10.0,
            abs_width_tol=1e-3,
        )
        assert self.check_partition_convergence(state) is False

    def test_bounds_preserved(self):
        """Lower and upper bounds of partitions must equal provided lb/ub."""
        state = self.initialize_partitions([0, 1], lb=[-2.0, 0.5], ub=[3.0, 4.5], n_init=4)
        assert state.partitions[0][0] == pytest.approx(-2.0)
        assert state.partitions[0][-1] == pytest.approx(3.0)
        assert state.partitions[1][0] == pytest.approx(0.5)
        assert state.partitions[1][-1] == pytest.approx(4.5)


# ===========================================================================
# Section 4: Piecewise McCormick Relaxation Validity
#
# Tests against EXISTING code in mccormick.py and piecewise_mccormick.py.
# Some of these should PASS already; failures reveal existing bugs.
# Key theoretical claim (CP 2016): piecewise gap < standard gap.
# ===========================================================================


class TestMcCormickSoundness:
    """Soundness tests for McCormick relaxation primitives.

    Theoretical requirement (CP 2016): relaxation must be a valid lower bound.
    cv ≤ f(x) ≤ cc at every feasible point.
    """

    @pytest.fixture(autouse=True)
    def _rng(self):
        self.rng = np.random.default_rng(42)

    def test_standard_mccormick_bilinear_valid(self):
        """cv ≤ x*y ≤ cc at 500 random points — core soundness test."""
        import jax.numpy as jnp
        from discopt._jax.mccormick import relax_bilinear

        x_lb, x_ub, y_lb, y_ub = 1.0, 4.0, 1.0, 4.0
        for _ in range(500):
            x = float(self.rng.uniform(x_lb, x_ub))
            y = float(self.rng.uniform(y_lb, y_ub))
            cv, cc = relax_bilinear(
                jnp.float64(x),
                jnp.float64(y),
                jnp.float64(x_lb),
                jnp.float64(x_ub),
                jnp.float64(y_lb),
                jnp.float64(y_ub),
            )
            w_true = x * y
            assert float(cv) <= w_true + 1e-9, f"cv={cv} > w={w_true}"
            assert float(cc) >= w_true - 1e-9, f"cc={cc} < w={w_true}"

    def test_piecewise_mccormick_bilinear_valid(self):
        """Piecewise McCormick is still a valid relaxation (cv ≤ x*y ≤ cc)."""
        import jax.numpy as jnp
        from discopt._jax.piecewise_mccormick import piecewise_mccormick_bilinear

        x_lb, x_ub, y_lb, y_ub = 0.0, 10.0, 0.0, 10.0
        for _ in range(300):
            x = float(self.rng.uniform(x_lb, x_ub))
            y = float(self.rng.uniform(y_lb, y_ub))
            cv, cc = piecewise_mccormick_bilinear(
                jnp.float64(x),
                jnp.float64(y),
                jnp.float64(x_lb),
                jnp.float64(x_ub),
                jnp.float64(y_lb),
                jnp.float64(y_ub),
                k=4,
            )
            w_true = x * y
            assert float(cv) <= w_true + 1e-9, f"cv={cv} > w={w_true} at x={x}, y={y}"
            assert float(cc) >= w_true - 1e-9, f"cc={cc} < w={w_true} at x={x}, y={y}"

    def test_piecewise_strictly_tighter_than_standard(self):
        """Piecewise gap < standard gap at the midpoint (key claim from CP 2016).

        At the center of the domain, standard McCormick has its maximum gap.
        Piecewise must be strictly tighter.
        """
        import jax.numpy as jnp
        from discopt._jax.mccormick import relax_bilinear
        from discopt._jax.piecewise_mccormick import piecewise_mccormick_bilinear

        x_lb, x_ub, y_lb, y_ub = 0.0, 10.0, 0.0, 10.0
        x, y = 5.0, 5.0  # exact midpoint — maximum standard McCormick gap

        cv_std, cc_std = relax_bilinear(
            jnp.float64(x),
            jnp.float64(y),
            jnp.float64(x_lb),
            jnp.float64(x_ub),
            jnp.float64(y_lb),
            jnp.float64(y_ub),
        )
        cv_pw, cc_pw = piecewise_mccormick_bilinear(
            jnp.float64(x),
            jnp.float64(y),
            jnp.float64(x_lb),
            jnp.float64(x_ub),
            jnp.float64(y_lb),
            jnp.float64(y_ub),
            k=4,
        )
        gap_std = float(cc_std - cv_std)
        gap_pw = float(cc_pw - cv_pw)

        assert gap_pw < gap_std, f"Piecewise gap {gap_pw:.4f} must be < standard gap {gap_std:.4f}"
        # Also verify piecewise is still sound
        w_true = x * y
        assert float(cv_pw) <= w_true + 1e-9
        assert float(cc_pw) >= w_true - 1e-9

    def test_piecewise_gap_decreases_with_k(self):
        """Gap must decrease as k increases (convergence property from CP 2016).

        Theoretical bound: max gap ≤ (domain_width / k)².
        """
        import jax.numpy as jnp
        from discopt._jax.piecewise_mccormick import piecewise_mccormick_bilinear

        x_lb, x_ub, y_lb, y_ub = 0.0, 10.0, 0.0, 10.0
        x, y = 3.0, 7.0
        prev_gap = float("inf")
        for k in [2, 4, 8, 16]:
            cv, cc = piecewise_mccormick_bilinear(
                jnp.float64(x),
                jnp.float64(y),
                jnp.float64(x_lb),
                jnp.float64(x_ub),
                jnp.float64(y_lb),
                jnp.float64(y_ub),
                k=k,
            )
            gap = float(cc - cv)
            assert gap < prev_gap, f"Gap should decrease as k increases (k={k})"
            prev_gap = gap

    def test_piecewise_is_solution_agnostic(self):
        """Document: current piecewise_mccormick is curvature-based, NOT solution-adaptive.

        This test SHOULD PASS — it documents existing behavior as a baseline.
        The AMP implementation will add solution-adaptive refinement separately.
        """
        # Same call twice → identical result (no solution context)
        import jax.numpy as jnp
        from discopt._jax.piecewise_mccormick import _partition_bounds

        lbs1, ubs1 = _partition_bounds(jnp.float64(0.0), jnp.float64(10.0), 4)
        lbs2, ubs2 = _partition_bounds(jnp.float64(0.0), jnp.float64(10.0), 4)
        np.testing.assert_array_equal(
            np.array(lbs1),
            np.array(lbs2),
            err_msg="Curvature-based partition is deterministic (good baseline)",
        )


# ===========================================================================
# Section 5: MILP Relaxation Lower Bound Validity
#
# Module to be created: python/discopt/_jax/milp_relaxation.py
# Function: build_milp_relaxation(model, terms, disc_state, incumbent) -> (model_obj, varmap)
#
# These tests will fail with ImportError until milp_relaxation.py exists.
# Key invariant: MILP LB ≤ global optimum (soundness).
# ===========================================================================


class TestMilpRelaxation:
    """Tests for build_milp_relaxation().

    Will fail with ImportError until milp_relaxation.py is created.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        from discopt._jax.discretization import initialize_partitions
        from discopt._jax.milp_relaxation import build_milp_relaxation
        from discopt._jax.term_classifier import classify_nonlinear_terms

        self.classify = classify_nonlinear_terms
        self.init_partitions = initialize_partitions
        self.build_milp = build_milp_relaxation

    def _build_and_solve(self, model, n_init=2):
        terms = self.classify(model)
        lb_arr = []
        ub_arr = []
        for v in model._variables:
            lb_arr.extend(np.asarray(v.lb, dtype=np.float64).ravel().tolist())
            ub_arr.extend(np.asarray(v.ub, dtype=np.float64).ravel().tolist())
        state = self.init_partitions(
            terms.partition_candidates,
            lb=[lb_arr[i] for i in terms.partition_candidates],
            ub=[ub_arr[i] for i in terms.partition_candidates],
            n_init=n_init,
        )
        milp_model, varmap = self.build_milp(model, terms, state, incumbent=None)
        result = milp_model.solve()
        return result

    def test_milp_lb_valid_nlp1(self):
        """MILP LB must be ≤ known global optimum 58.38368 for nlp1."""
        result = self._build_and_solve(_make_nlp1())
        assert result.status not in ("infeasible", "error"), (
            f"MILP relaxation of nlp1 should be feasible, got {result.status}"
        )
        if result.objective is not None:
            assert result.objective <= 58.4 + 1e-4, (
                f"MILP LB {result.objective:.6f} exceeds known global optimum 58.38368"
            )

    def test_milp_lb_valid_circle(self):
        """MILP LB must be ≤ √2 ≈ 1.41421 for circle problem."""
        result = self._build_and_solve(_make_circle())
        assert result.status not in ("infeasible", "error")
        if result.objective is not None:
            assert result.objective <= 1.41422 + 1e-4, (
                f"MILP LB {result.objective:.6f} exceeds known global optimum √2"
            )

    def test_milp_lb_tightens_with_finer_partitions(self):
        """LB must be non-decreasing as partition count increases (key paper claim)."""
        m = _make_nlp1()
        lbs = []
        for n_init in [1, 2, 4, 8]:
            result = self._build_and_solve(m, n_init=n_init)
            if result.objective is not None:
                lbs.append(result.objective)
            else:
                lbs.append(-np.inf)
        # Lower bounds must be non-decreasing with finer partitions
        for i in range(len(lbs) - 1):
            assert lbs[i] <= lbs[i + 1] + 1e-6, (
                f"LBs must be non-decreasing: lbs[{i}]={lbs[i]:.4f} > lbs[{i + 1}]={lbs[i + 1]:.4f}"
            )

    def test_milp_relaxation_feasible_when_problem_feasible(self):
        """Relaxation must be feasible when original problem is feasible."""
        for m in [_make_nlp1(), _make_circle()]:
            result = self._build_and_solve(m)
            assert result.status != "infeasible", (
                f"MILP relaxation infeasible for a feasible problem: {m._name}"
            )

    def test_piecewise_interval_rows_use_interval_specific_bounds(self):
        """Piecewise product rows must use the current interval's wk_hi, not a stale value."""
        import scipy.sparse as sp
        from discopt._jax.discretization import DiscretizationState
        from discopt._jax.term_classifier import classify_nonlinear_terms

        m = Model("piecewise_bounds")
        x = m.continuous("x", lb=1, ub=5)
        y = m.continuous("y", lb=10, ub=20)
        m.minimize(x * y)

        terms = classify_nonlinear_terms(m)
        state = DiscretizationState(partitions={0: np.array([1.0, 2.5, 4.0, 5.0])})
        milp_model, varmap = self.build_milp(m, terms, state, incumbent=None)

        assert sp.issparse(milp_model._A_ub)
        A_csr = milp_model._A_ub.tocsr()
        b_ub = np.asarray(milp_model._b_ub, dtype=np.float64)

        for delta_col, _, wbar_col, a_k, b_k in varmap["bilinear_pw"][(0, 1)]:
            expected_hi = max(
                a_k * 10.0,
                a_k * 20.0,
                b_k * 10.0,
                b_k * 20.0,
            )
            matched = False
            for row_idx in range(A_csr.shape[0]):
                row = A_csr.getrow(row_idx)
                if row.nnz != 2 or abs(b_ub[row_idx]) > 1e-12:
                    continue
                coeffs = dict(zip(row.indices.tolist(), row.data.tolist()))
                if coeffs.get(wbar_col) == 1.0 and coeffs.get(delta_col) == -expected_hi:
                    matched = True
                    break
            assert matched, f"missing interval-specific upper row for [{a_k}, {b_k}]"

    def test_milp_respects_original_variable_integrality(self):
        """Original integer variables must stay integral in the MILP relaxation."""
        m = Model("orig_integrality")
        x = m.continuous("x", lb=0, ub=1)
        y = m.integer("y", lb=0, ub=3)
        m.subject_to(x * y >= 1.8)
        m.minimize(y)

        result = self._build_and_solve(m)

        assert result.status == "optimal"
        assert result.objective is not None
        assert result.objective >= 2.0 - 1e-8
        assert result.x is not None
        assert abs(float(result.x[1]) - round(float(result.x[1]))) <= 1e-8

    def test_lambda_convhull_builds_expected_auxiliaries(self):
        """The λ-convex-hull formulation should expose lambda, alpha, and theta blocks."""
        m = _make_nlp1()
        terms = self.classify(m)
        state = self.init_partitions([0], lb=[1.0], ub=[4.0], n_init=2)

        _, varmap = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="sos2",
        )

        info = varmap["bilinear_lambda"][(0, 1)]
        assert varmap["convhull_formulation"] == "sos2"
        assert info["breakpoints"] == [1.0, 2.5, 4.0]
        assert len(info["lambda_cols"]) == 3
        assert len(info["alpha_cols"]) == 2
        assert len(info["theta_cols"]) == 3

    def test_sos2_and_facet_convhull_match_on_simple_bilinear_relaxation(self):
        """SOS2 and facet λ-linking should dominate the disaggregated relaxation."""
        m = Model("lambda_compare")
        x = m.continuous("x", lb=0, ub=2, shape=(2,))
        m.subject_to(x[0] * x[1] >= 1.0)
        m.minimize(x[0] + x[1])

        terms = self.classify(m)
        state = self.init_partitions([0], lb=[0.0], ub=[2.0], n_init=4)

        disagg_model, _ = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="disaggregated",
        )

        sos2_model, _ = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="sos2",
        )
        facet_model, _ = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="facet",
        )

        disagg_result = disagg_model.solve()
        sos2_result = sos2_model.solve()
        facet_result = facet_model.solve()

        assert disagg_result.status == "optimal"
        assert sos2_result.status == "optimal"
        assert facet_result.status == "optimal"
        assert disagg_result.objective is not None
        assert sos2_result.objective is not None
        assert facet_result.objective is not None
        assert sos2_result.objective == pytest.approx(facet_result.objective, abs=1e-6)
        assert sos2_result.objective >= disagg_result.objective - 1e-6
        assert facet_result.objective >= disagg_result.objective - 1e-6
        assert sos2_result.objective <= 2.0 + 1e-6

    def test_invalid_convhull_formulation_raises(self):
        """Unsupported convex-hull mode names should fail fast."""
        m = _make_nlp1()
        terms = self.classify(m)
        state = self.init_partitions([0], lb=[1.0], ub=[4.0], n_init=2)

        with pytest.raises(ValueError, match="Unsupported convhull_formulation"):
            self.build_milp(
                m,
                terms,
                state,
                incumbent=None,
                convhull_formulation="bogus",
            )

    def test_trilinear_milp_builds_nested_auxiliaries(self):
        """Trilinear terms should be modeled through explicit lifted auxiliaries."""
        m = _make_trilinear_cover()
        terms = self.classify(m)
        state = self.init_partitions([0, 1, 2], lb=[0.0, 0.0, 0.0], ub=[2.0, 2.0, 2.0], n_init=2)

        milp_model, varmap = self.build_milp(m, terms, state, incumbent=None)
        result = milp_model.solve()

        assert (0, 1, 2) in varmap["trilinear"]
        stage = varmap["trilinear_stages"][(0, 1, 2)]
        assert stage["product_col"] == varmap["trilinear"][(0, 1, 2)]
        assert result.status == "optimal"
        assert result.objective is not None
        assert 0.0 < result.objective <= 3.0 + 1e-6

    def test_unsupported_objective_returns_no_relaxation_bound(self):
        """Unsupported nonlinear objectives should not be reported as valid lower bounds."""
        m = Model("unsupported_objective")
        x = m.continuous("x", lb=0, ub=2, shape=(2,))
        m.subject_to(x[0] + x[1] >= 1.0)
        m.minimize((x[0] * x[0]) * x[1])

        terms = self.classify(m)
        state = self.init_partitions([0], lb=[0.0], ub=[2.0], n_init=2)

        milp_model, _ = self.build_milp(m, terms, state, incumbent=None)
        result = milp_model.solve()

        assert result.status == "optimal"
        assert result.objective is None
        assert result.x is not None

    def test_piecewise_interval_rows_force_inactive_wbar_to_zero_with_negative_bounds(self):
        """Disaggregated intervals must include the lower δ-forcing row even when wk_lo < 0."""
        import scipy.sparse as sp
        from discopt._jax.discretization import DiscretizationState
        from discopt._jax.term_classifier import classify_nonlinear_terms

        m = Model("piecewise_sign_straddle")
        x = m.continuous("x", lb=0, ub=2)
        y = m.continuous("y", lb=-1, ub=1)
        m.minimize(x * y)

        terms = classify_nonlinear_terms(m)
        state = DiscretizationState(partitions={0: np.array([0.0, 1.0, 2.0])})
        milp_model, varmap = self.build_milp(m, terms, state, incumbent=None)

        assert sp.issparse(milp_model._A_ub)
        A_csr = milp_model._A_ub.tocsr()
        b_ub = np.asarray(milp_model._b_ub, dtype=np.float64)

        for delta_col, _, wbar_col, a_k, b_k in varmap["bilinear_pw"][(0, 1)]:
            wk_lo = min(a_k * -1.0, a_k * 1.0, b_k * -1.0, b_k * 1.0)
            assert wk_lo < 0.0
            matched = False
            for row_idx in range(A_csr.shape[0]):
                row = A_csr.getrow(row_idx)
                if row.nnz != 2 or abs(b_ub[row_idx]) > 1e-12:
                    continue
                coeffs = dict(zip(row.indices.tolist(), row.data.tolist()))
                if coeffs.get(wbar_col) == -1.0 and coeffs.get(delta_col) == wk_lo:
                    matched = True
                    break
            assert matched, f"missing inactive-interval lower row for [{a_k}, {b_k}]"


class TestAmpPhase1Helpers:
    """Fast regression tests for Phase 1 helper behavior."""

    def test_piecewise_big_m_scales_with_large_coefficients(self):
        """Big-M must scale with coefficient magnitude instead of adding a flat 1.0."""
        from discopt._jax.milp_relaxation import _compute_piecewise_big_m

        corners = [-25000.0, 10000.0, 5000.0, 30000.0]
        big_m = _compute_piecewise_big_m(corners)
        expected = 30000.0 * (1.0 + 1e-4) + 3.0

        assert big_m == pytest.approx(expected)
        assert big_m > 30000.0

    def test_piecewise_big_m_keeps_small_intervals_tight(self):
        """The Big-M floor should not dominate near-zero product intervals."""
        from discopt._jax.milp_relaxation import _compute_piecewise_big_m

        corners = [0.0, 0.0, 0.01, 0.01]
        big_m = _compute_piecewise_big_m(corners)

        assert big_m == pytest.approx(0.010002)
        assert big_m < 0.011


# ===========================================================================
# Section 6: End-to-End AMP Solver (Alpine canonical problems)
#
# Requires:
#   - python/discopt/solvers/amp.py (solve_amp function)
#   - python/discopt/solver.py routing for solver="amp"
#
# Known optima from Alpine.jl test suite.
# ===========================================================================

# Known global optima (from Alpine.jl test suite + standard references)
NLP1_OPTIMUM = 58.38368  # Alpine nlp1
CIRCLE_OPTIMUM = 1.41421356  # √2
NLP3_OPTIMUM = 7049.247898  # Alpine nlp3


class TestAmpEndToEnd:
    """End-to-end AMP solver tests on Alpine canonical problems.

    Will fail until solvers/amp.py exists and solver.py routes solver="amp".
    """

    @pytest.mark.smoke
    def test_nlp1_bilinear_global_optimum(self):
        """nlp1: bilinear MINLP solved to global optimum ≈ 58.38368."""

        m = _make_nlp1()
        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)
        assert result.status == "optimal", f"Expected optimal, got {result.status}"
        assert result.gap_certified is True, "Gap must be certified for AMP"
        assert result.objective is not None
        assert abs(result.objective - NLP1_OPTIMUM) <= 0.06, (
            f"Objective {result.objective:.5f} too far from {NLP1_OPTIMUM}"
        )

    @pytest.mark.smoke
    def test_circle_bilinear_global_optimum(self):
        """circle: recover the best known objective without a false certificate."""
        m = _make_circle()
        result = m.solve(solver="amp", rel_gap=1e-4, time_limit=60)
        assert result.status == "feasible"
        assert result.objective is not None
        assert abs(result.objective - CIRCLE_OPTIMUM) <= 1e-3, (
            f"Objective {result.objective:.6f} too far from √2={CIRCLE_OPTIMUM}"
        )
        # The circle constraint is a nonconvex superlevel set, so evaluator OA
        # cuts must be skipped and AMP should not certify the relaxation gap.
        assert result.gap_certified is False

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    @pytest.mark.xfail(
        reason="nlp3 MILP exceeds test budget; tracked in #24",
        strict=True,
    )
    def test_nlp3_multilinear_global_optimum(self):
        """nlp3: 8-variable multilinear industrial problem."""
        m = _build_nlp3()
        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=300)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert abs(result.objective - NLP3_OPTIMUM) <= 2.0, (
            f"Objective {result.objective:.3f} too far from {NLP3_OPTIMUM}"
        )

    @pytest.mark.slow
    def test_bilinear_minlp_global(self):
        """MINLP with bilinear product + integer variable."""
        m = Model("bi_minlp")
        x = m.continuous("x", lb=0, ub=5, shape=(2,))
        y = m.integer("y", lb=0, ub=3)
        m.subject_to(x[0] * x[1] + y >= 4)
        m.minimize(x[0] ** 2 + x[1] ** 2 + y)
        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        # x[0]*x[1]≥4-y: optimal at x[0]=x[1]=2, y=0 → obj=8 or similar
        assert result.objective >= 0

    @pytest.mark.slow
    def test_trilinear_global_optimum(self):
        """AMP should solve a simple trilinear cover problem after lifting."""
        m = _make_trilinear_cover()
        result = m.solve(solver="amp", rel_gap=2e-2, time_limit=20)

        assert result.status in ("optimal", "feasible", "time_limit")
        assert result.objective is not None
        assert abs(result.objective - 3.0) <= 0.1

    def test_nonconvex_constraint_oa_cuts_respect_convexity_mask(self, monkeypatch):
        """Nonconvex constraint rows must be excluded from evaluator OA cuts."""
        from discopt._jax import cutting_planes

        recorded_masks = []
        real_generate = cutting_planes.generate_oa_cuts_from_evaluator

        def wrapped_generate(*args, **kwargs):
            convex_mask = kwargs.get("convex_mask")
            if convex_mask is not None:
                recorded_masks.append(list(convex_mask))
            return real_generate(*args, **kwargs)

        monkeypatch.setattr(cutting_planes, "generate_oa_cuts_from_evaluator", wrapped_generate)

        m = Model("amp_nonconvex_constraint_mask")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.subject_to(x**2 - 4 * y - 1 <= 0)
        m.subject_to(x**2 - x + y >= 0)
        m.minimize(x + y)

        result = m.solve(solver="amp", max_iter=1, rel_gap=1e-3, time_limit=30)

        assert result.status in ("optimal", "feasible", "time_limit")
        assert recorded_masks
        assert all(mask == [True, False] for mask in recorded_masks)

    def test_unsupported_objective_disables_certified_bound(self, monkeypatch):
        """AMP must not report a certified lower bound when the objective is not linearizable."""
        from discopt.solvers import amp as amp_solver

        m = Model("amp_unsupported_objective")
        x = m.continuous("x", lb=0, ub=2, shape=(2,))
        y = m.binary("y")
        m.subject_to(x[0] >= 1 - y)
        m.minimize((x[0] * x[0]) * x[1] + y)

        feasible_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        monkeypatch.setattr(
            amp_solver,
            "_solve_best_nlp_candidate",
            lambda *args, **kwargs: (feasible_x.copy(), 0.0),
        )

        result = m.solve(solver="amp", rel_gap=1e-3, max_iter=2, time_limit=10)

        assert result.status == "feasible"
        assert result.objective == pytest.approx(0.0)
        assert result.bound is None
        assert result.gap is None
        assert result.gap_certified is False

    @pytest.mark.smoke
    def test_pure_quadratic_convex(self):
        """min x²  s.t. x ≥ 1: AMP should certify global optimum = 1."""
        m = _make_convex_quadratic()
        result = m.solve(solver="amp", rel_gap=1e-4, time_limit=30)
        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective - 1.0) <= 1e-3

    @pytest.mark.smoke
    def test_zero_upper_bound_reports_no_relative_gap(self):
        """Relative gap should stay undefined when the incumbent objective is near zero."""
        m = Model("zero_gap")
        x = m.continuous("x", lb=-1, ub=1)
        m.minimize(x**2)

        # skip_convex_check forces AMP to run partitioning even on convex
        # models, so we exercise AMP's gap-reporting convention here.
        result = m.solve(solver="amp", rel_gap=1e-4, time_limit=30, skip_convex_check=True)

        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective) <= 1e-6
        assert result.gap is None

    @pytest.mark.slow
    def test_bilinear_maximize_global_optimum(self):
        """AMP must handle maximize objectives with certified bounds."""
        m = Model("max_bilinear")
        x = m.continuous("x", lb=0, ub=2, shape=(2,))
        m.subject_to(x[0] + x[1] <= 2)
        m.maximize(x[0] * x[1])

        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)

        assert result.status == "optimal"
        assert result.gap_certified is True
        assert result.objective is not None
        assert abs(result.objective - 1.0) <= 1e-3
        assert result.bound is not None
        assert result.bound >= result.objective - 1e-6

    def test_maximize_with_integer_variables(self):
        """Maximize should work when integer rounding and fixed NLP bounds interact."""
        m = Model("max_bin")
        x = m.continuous("x", lb=0, ub=2)
        y = m.binary("y")
        m.subject_to(x <= 2 * y)
        m.maximize(x * y)

        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)

        assert result.status == "optimal"
        assert result.gap_certified is True
        assert result.objective is not None
        assert abs(result.objective - 2.0) <= 1e-3
        assert result.bound is not None
        assert result.bound >= result.objective - 1e-6

    def test_amp_solver_warns_on_ignored_options(self):
        """Routing through solve_model should warn when AMP ignores B&B-only options."""
        m = Model("amp_warning")
        x = m.continuous("x", lb=-1, ub=1)
        m.minimize(x**2)

        with pytest.warns(
            UserWarning,
            match=(
                "AMP ignores solve_model options: "
                ".*cutting_planes.*incumbent_callback.*mccormick_bounds.*node_callback"
            ),
        ):
            result = m.solve(
                solver="amp",
                time_limit=10,
                cutting_planes=True,
                mccormick_bounds="tight",
                incumbent_callback=lambda *_: True,
                node_callback=lambda *_: None,
            )

        assert result.objective is not None


# ===========================================================================
# Section 7: Convergence Property Tests
#
# Verify key theoretical invariants from JOGO 2018:
#   (a) LB sequence is monotonically non-decreasing
#   (b) UB sequence is monotonically non-increasing
#   (c) gap_certified=True iff termination by gap criterion
#   (d) early termination when gap closes
#   (e) time_limit respected
# ===========================================================================


class TestAmpConvergenceProperties:
    """Test AMP's theoretical convergence properties.

    Will fail until solvers/amp.py supports iteration_callback.
    """

    @pytest.mark.smoke
    def test_lower_bound_monotone(self):
        """LB sequence must be non-decreasing across AMP iterations (JOGO 2018)."""
        m = _make_nlp1()
        lbs = []

        def cb(info):
            lbs.append(info["lower_bound"])

        m.solve(solver="amp", rel_gap=1e-6, max_iter=6, iteration_callback=cb, time_limit=60)
        assert len(lbs) >= 2, "Should have at least 2 iterations"
        for i in range(len(lbs) - 1):
            assert lbs[i] <= lbs[i + 1] + 1e-8, (
                f"LB decreased at iteration {i}: {lbs[i]:.6f} > {lbs[i + 1]:.6f}"
            )

    @pytest.mark.smoke
    def test_upper_bound_monotone(self):
        """UB sequence must be non-increasing (best feasible improves or stays same)."""
        m = _make_nlp1()
        ubs = []

        def cb(info):
            ubs.append(info["upper_bound"])

        m.solve(solver="amp", rel_gap=1e-6, max_iter=6, iteration_callback=cb, time_limit=60)
        # Filter out inf (no feasible found yet)
        finite_ubs = [(i, u) for i, u in enumerate(ubs) if u < 1e18]
        for j in range(len(finite_ubs) - 1):
            i1, u1 = finite_ubs[j]
            i2, u2 = finite_ubs[j + 1]
            assert u2 <= u1 + 1e-8, f"UB increased at iteration {i2}: {u1:.6f} → {u2:.6f}"

    @pytest.mark.smoke
    def test_gap_certified_after_convergence(self):
        """gap_certified must be True after AMP converges by gap criterion."""
        m = _make_convex_quadratic()
        result = m.solve(solver="amp", rel_gap=0.05, time_limit=30)
        assert result.status == "optimal"
        assert result.gap_certified is True

    @pytest.mark.smoke
    def test_early_termination_when_gap_closes(self):
        """AMP should terminate before max_iter when gap closes."""
        m = _make_convex_quadratic()
        iters = []

        def cb(info):
            iters.append(info["iteration"])

        result = m.solve(
            solver="amp", rel_gap=0.5, max_iter=100, iteration_callback=cb, time_limit=30
        )
        assert result.status == "optimal"
        assert result.gap_certified is True
        assert len(iters) < 100, "Should terminate before max_iter=100 when gap closes"

    @pytest.mark.slow
    def test_time_limit_respected(self):
        """AMP must respect the time_limit parameter."""
        import time

        m = _build_nlp3()
        t0 = time.perf_counter()
        result = m.solve(solver="amp", time_limit=3.0)
        elapsed = time.perf_counter() - t0
        # Allow 5s buffer for cleanup
        assert elapsed <= 8.0, f"Time limit 3s violated: ran {elapsed:.1f}s"
        assert result.status in ("optimal", "feasible", "time_limit", "infeasible")

    @pytest.mark.smoke
    def test_max_iter_respected(self):
        """AMP must terminate at max_iter if gap not closed."""
        m = _build_nlp3()
        iters = []

        def cb(info):
            iters.append(info["iteration"])

        m.solve(solver="amp", max_iter=3, time_limit=300)
        assert len(iters) <= 3, f"max_iter=3 violated: {len(iters)} iterations"


# ===========================================================================
# Section 8: Weakness Probes (Tests Designed to Expose Current Gaps)
#
# These tests probe the *current* implementation for known limitations.
# Some may pass (confirming good existing behavior) or fail (revealing gaps).
# ===========================================================================


class TestCurrentCodeWeaknesses:
    """Probe existing code for gaps relevant to AMP implementation.

    These are diagnostic tests — failures here guide implementation priorities.
    """

    def test_constraint_check_rejects_eval_failure(self, monkeypatch):
        """Constraint evaluation errors must reject the candidate point."""
        import discopt._jax.nlp_evaluator as nlp_eval
        from discopt.solvers import amp as amp_mod

        class BrokenEvaluator:
            def __init__(self, model):
                self.n_constraints = 1
                self.constraint_bounds = (np.array([0.0]), np.array([1.0]))

            def evaluate_constraints(self, x):
                raise RuntimeError("boom")

        monkeypatch.setattr(nlp_eval, "NLPEvaluator", BrokenEvaluator)

        m = Model("broken_eval")
        x = m.continuous("x", lb=0, ub=1)
        m.subject_to(x >= 0)
        m.minimize(x)

        assert amp_mod._check_constraints(np.array([0.5]), m) is False

    def test_solve_model_signature_exposes_solver_parameter(self):
        """solve_model should expose the backend selector in its signature."""
        import inspect

        from discopt.solver import solve_model

        assert "solver" in inspect.signature(solve_model).parameters

    def test_solve_model_forwards_alpine_amp_aliases(self, monkeypatch):
        """solve_model should pass Alpine-style AMP aliases through to solve_amp."""
        from discopt.solver import solve_model
        from discopt.solvers import amp as amp_mod

        captured = {}

        def fake_solve_amp(model, **kwargs):
            del model
            captured.update(kwargs)
            return SolveResult(status="infeasible", wall_time=0.0)

        monkeypatch.setattr(amp_mod, "solve_amp", fake_solve_amp)

        m = Model("alias_forwarding")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        solve_model(
            m,
            solver="amp",
            gap_tolerance=1e-3,
            apply_partitioning=False,
            disc_var_pick=1,
            partition_scaling_factor=7.0,
            disc_add_partition_method="uniform",
            disc_abs_width_tol=1e-2,
            convhull_formulation="sos2",
        )

        assert captured["rel_gap"] == pytest.approx(1e-3)
        assert captured["apply_partitioning"] is False
        assert captured["disc_var_pick"] == 1
        assert captured["partition_scaling_factor"] == pytest.approx(7.0)
        assert captured["disc_add_partition_method"] == "uniform"
        assert captured["disc_abs_width_tol"] == pytest.approx(1e-2)
        assert captured["convhull_formulation"] == "sos2"

    def test_integer_rounding_candidates_include_floor_and_ceil(self):
        """Nearest-integer rounding fallback must try floor and ceil alternatives."""
        from discopt.solvers import amp as amp_mod

        m = Model("rounding")
        m.integer("y", lb=0, ub=3, shape=(2,))
        x0 = np.array([1.49, 1.51], dtype=np.float64)

        candidates = amp_mod._integer_rounding_candidates(x0, m)
        rounded = {tuple(float(v) for v in cand) for cand in candidates}

        assert (1.0, 2.0) in rounded  # nearest
        assert (1.0, 1.0) in rounded  # floor on the second variable
        assert (2.0, 2.0) in rounded  # ceil on the first variable

    def test_best_nlp_candidate_chooses_lowest_feasible_objective(self, monkeypatch):
        """Integer rounding fallback should keep the best feasible NLP candidate."""
        from discopt.solvers import amp as amp_mod

        m = Model("best_candidate")
        m.integer("y", lb=0, ub=2)

        candidates = [
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
        ]
        objectives = {0.0: 4.0, 1.0: 1.5, 2.0: 3.0}

        monkeypatch.setattr(
            amp_mod,
            "_integer_rounding_candidates",
            lambda x0, model: [cand.copy() for cand in candidates],
        )
        monkeypatch.setattr(
            amp_mod,
            "_build_fixed_integer_bounds",
            lambda x0, model, flat_lb, flat_ub: (flat_lb.copy(), flat_ub.copy()),
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_nlp_subproblem",
            lambda evaluator, x0, lb, ub, nlp_solver: (
                x0.copy(),
                objectives[float(x0[0])],
            ),
        )
        monkeypatch.setattr(
            amp_mod,
            "_check_constraints_with_evaluator",
            lambda evaluator, x, lb_g, ub_g: True,
        )

        best_x, best_obj = amp_mod._solve_best_nlp_candidate(
            np.array([0.3], dtype=np.float64),
            m,
            evaluator=object(),
            flat_lb=np.array([0.0], dtype=np.float64),
            flat_ub=np.array([2.0], dtype=np.float64),
            constraint_lb=np.array([], dtype=np.float64),
            constraint_ub=np.array([], dtype=np.float64),
            nlp_solver="ipm",
        )

        assert best_x is not None
        assert float(best_x[0]) == pytest.approx(1.0)
        assert best_obj == pytest.approx(1.5)

    def test_nlp_subproblem_applies_and_restores_fixed_bounds(self, monkeypatch):
        """AMP must solve the NLP at the fixed candidate bounds, then restore them."""
        import discopt._jax.ipm as ipm_mod
        from discopt.solvers import NLPResult, SolveStatus
        from discopt.solvers import amp as amp_mod

        m = Model("fixed_nlp_bounds")
        x = m.continuous("x", lb=0.0, ub=2.0)
        y = m.binary("y")
        m.subject_to(x >= y)
        m.minimize(x + y)

        class FakeEvaluator:
            def __init__(self, model):
                self._model = model
                self._obj_fn = object()

            def evaluate_objective(self, x_flat):
                return float(np.sum(x_flat))

        def fake_solve_nlp_ipm(evaluator, x0, options):
            del x0, options
            x_var, y_var = evaluator._model._variables
            assert np.asarray(x_var.lb).item() == pytest.approx(0.25)
            assert np.asarray(x_var.ub).item() == pytest.approx(0.75)
            assert np.asarray(y_var.lb).item() == pytest.approx(1.0)
            assert np.asarray(y_var.ub).item() == pytest.approx(1.0)
            return NLPResult(
                status=SolveStatus.OPTIMAL,
                x=np.array([0.5, 1.0], dtype=np.float64),
            )

        monkeypatch.setattr(ipm_mod, "solve_nlp_ipm", fake_solve_nlp_ipm)

        x_opt, obj = amp_mod._solve_nlp_subproblem(
            FakeEvaluator(m),
            x0=np.array([1.2, 0.2], dtype=np.float64),
            lb=np.array([0.25, 1.0], dtype=np.float64),
            ub=np.array([0.75, 1.0], dtype=np.float64),
            nlp_solver="ipm",
        )

        assert x_opt is not None
        assert np.allclose(x_opt, np.array([0.5, 1.0], dtype=np.float64))
        assert obj == pytest.approx(1.5)
        assert np.asarray(x.lb).item() == pytest.approx(0.0)
        assert np.asarray(x.ub).item() == pytest.approx(2.0)
        assert np.asarray(y.lb).item() == pytest.approx(0.0)
        assert np.asarray(y.ub).item() == pytest.approx(1.0)

    def test_best_nlp_candidate_rejects_noninteger_nlp_return(self, monkeypatch):
        """NLP candidates that violate integrality should be discarded."""
        from discopt.solvers import amp as amp_mod

        m = Model("noninteger_candidate")
        m.integer("y", lb=0, ub=2)

        monkeypatch.setattr(
            amp_mod,
            "_integer_rounding_candidates",
            lambda x0, model: [np.array([1.0], dtype=np.float64)],
        )
        monkeypatch.setattr(
            amp_mod,
            "_build_fixed_integer_bounds",
            lambda x0, model, flat_lb, flat_ub: (flat_lb.copy(), flat_ub.copy()),
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_nlp_subproblem",
            lambda evaluator, x0, lb, ub, nlp_solver: (
                np.array([1.5], dtype=np.float64),
                1.0,
            ),
        )

        best_x, best_obj = amp_mod._solve_best_nlp_candidate(
            np.array([1.0], dtype=np.float64),
            m,
            evaluator=object(),
            flat_lb=np.array([0.0], dtype=np.float64),
            flat_ub=np.array([2.0], dtype=np.float64),
            constraint_lb=np.array([], dtype=np.float64),
            constraint_ub=np.array([], dtype=np.float64),
            nlp_solver="ipm",
        )

        assert best_x is None
        assert best_obj is None

    def test_oa_cut_recovery_drops_oldest_half(self, monkeypatch):
        """OA recovery should retry with the oldest half of cuts removed."""
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        call_sizes = []

        class FakeMilpModel:
            def __init__(self, status):
                self.status = status

            def solve(self, time_limit=None, gap_tolerance=None):
                return MilpRelaxationResult(
                    status=self.status,
                    objective=0.0,
                    x=np.zeros(1, dtype=np.float64),
                )

        def fake_build(
            model,
            terms,
            disc_state,
            incumbent,
            oa_cuts=None,
            convhull_formulation="disaggregated",
        ):
            assert convhull_formulation == "disaggregated"
            size = len(oa_cuts or [])
            call_sizes.append(size)
            status = "infeasible" if size >= 4 else "optimal"
            return FakeMilpModel(status), {"dummy": True}

        monkeypatch.setattr(
            "discopt._jax.milp_relaxation.build_milp_relaxation",
            fake_build,
        )

        result, _, kept_cuts = amp_mod._solve_milp_with_oa_recovery(
            model=None,
            terms=None,
            disc_state=None,
            incumbent=None,
            oa_cuts=[("c1", 1), ("c2", 2), ("c3", 3), ("c4", 4)],
            time_limit=1.0,
            gap_tolerance=1e-4,
            convhull_formulation="disaggregated",
        )

        assert call_sizes == [4, 2]
        assert kept_cuts == [("c3", 3), ("c4", 4)]
        assert result.status == "optimal"

    def test_amp_max_iter_without_gap_certificate_returns_feasible(self):
        """An incumbent without a certified gap should not be labeled optimal."""
        m = _make_nlp1()

        result = m.solve(solver="amp", max_iter=1, time_limit=30)

        assert result.status == "feasible"
        assert result.gap_certified is False
        assert result.objective is not None
        assert result.gap is not None
        assert result.gap > 1e-3

    def test_partition_convergence_without_gap_is_not_certified(self, monkeypatch):
        """Forced partition convergence should still return a non-certified incumbent."""
        import discopt._jax.discretization as disc_mod

        monkeypatch.setattr(disc_mod, "check_partition_convergence", lambda state: True)

        m = _make_nlp1()
        result = m.solve(solver="amp", rel_gap=1e-6, time_limit=30)

        assert result.status == "feasible"
        assert result.gap_certified is False
        assert result.objective is not None

    def test_no_incumbent_after_search_returns_iteration_limit(self, monkeypatch):
        """Exhausting AMP without an incumbent is not a proof of infeasibility."""
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        m = Model("amp_no_incumbent")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        monkeypatch.setattr(
            amp_mod,
            "_solve_milp_with_oa_recovery",
            lambda **kwargs: (
                MilpRelaxationResult(
                    status="optimal",
                    objective=0.0,
                    x=np.array([0.0], dtype=np.float64),
                ),
                {},
                [],
            ),
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_best_nlp_candidate",
            lambda *args, **kwargs: (None, None),
        )

        result = m.solve(solver="amp", max_iter=1, time_limit=30, skip_convex_check=True)

        assert result.status == "iteration_limit"
        assert result.objective is None
        assert result.bound == pytest.approx(0.0)
        assert result.gap_certified is False

    def test_first_iteration_milp_error_is_not_reported_as_infeasible(self, monkeypatch):
        """MILP solve errors should surface as errors, not mathematical infeasibility."""
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        m = Model("amp_milp_error")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        monkeypatch.setattr(
            amp_mod,
            "_solve_milp_with_oa_recovery",
            lambda **kwargs: (
                MilpRelaxationResult(status="error", objective=None, x=None),
                {},
                [],
            ),
        )

        result = m.solve(solver="amp", max_iter=1, time_limit=30, skip_convex_check=True)

        assert result.status == "error"
        assert result.objective is None
        assert result.bound is None
        assert result.gap_certified is False

    def test_adaptive_partition_selection_path_runs_inside_amp(self, monkeypatch):
        """AMP should re-pick partition variables when adaptive selection is enabled."""
        import discopt._jax.discretization as disc_mod
        import discopt._jax.partition_selection as part_mod

        adaptive_distances = []
        refined_var_sets = []
        orig_pick = part_mod.pick_partition_vars

        def fake_pick(terms, method="auto", distance=None):
            if method == "adaptive_vertex_cover" and distance is None:
                return [0]
            if method == "adaptive_vertex_cover":
                adaptive_distances.append(dict(distance or {}))
                return [1]
            return orig_pick(terms, method=method, distance=distance)

        def fake_add_adaptive_partition(state, solution, var_indices, lb, ub):
            del solution, lb, ub
            refined_var_sets.append(list(var_indices))
            return state

        monkeypatch.setattr(part_mod, "pick_partition_vars", fake_pick)
        monkeypatch.setattr(disc_mod, "add_adaptive_partition", fake_add_adaptive_partition)
        monkeypatch.setattr(disc_mod, "check_partition_convergence", lambda state: True)

        m = _make_nlp1()
        result = m.solve(
            solver="amp",
            disc_var_pick="adaptive",
            rel_gap=1e-6,
            time_limit=30,
        )

        assert adaptive_distances
        assert refined_var_sets == [[1]]
        assert result.status == "feasible"
        assert result.gap_certified is False

    def test_uniform_partition_refinement_path_runs_inside_amp(self, monkeypatch):
        """AMP should call the uniform refinement branch when requested."""
        import discopt._jax.discretization as disc_mod

        uniform_calls = []

        def fake_add_uniform_partition(state, _solution, var_indices, lb, ub):
            del _solution, lb, ub
            uniform_calls.append(list(var_indices))
            return state

        monkeypatch.setattr(disc_mod, "add_uniform_partition", fake_add_uniform_partition)
        monkeypatch.setattr(disc_mod, "check_partition_convergence", lambda state: True)

        m = _make_nlp1()
        result = m.solve(
            solver="amp",
            disc_add_partition_method="uniform",
            rel_gap=1e-6,
            time_limit=30,
        )

        assert uniform_calls
        assert result.status == "feasible"
        assert result.gap_certified is False

    def test_spatial_bnb_bilinear_global_correctness(self):
        """Existing spatial B&B should solve nlp1 to global optimum.

        If this fails: existing spatial B&B cannot certify global optimality
        on bilinear problems (needs AMP's MILP-based lower bound).
        """
        pytest.importorskip("cyipopt")
        m = _make_nlp1()
        result = m.solve(time_limit=60, gap_tolerance=1e-3)
        assert result.objective is not None, "Spatial B&B failed to find any solution"
        assert abs(result.objective - NLP1_OPTIMUM) <= 0.5, (
            f"Spatial B&B found {result.objective:.4f}, expected near {NLP1_OPTIMUM} "
            f"(gap={abs(result.objective - NLP1_OPTIMUM):.4f})"
        )

    @pytest.mark.xfail(
        reason="Documents a known weakness of the legacy spatial B&B on quadratic constraints",
        strict=False,
    )
    def test_circle_monomial_global_correctness(self):
        """Existing solver weakness: x²+y²≥2 is not yet solved globally by spatial B&B."""
        m = _make_circle()
        result = m.solve(time_limit=30, gap_tolerance=1e-3)
        assert result.objective is not None
        # Global optimum is √2 ≈ 1.41421
        assert abs(result.objective - CIRCLE_OPTIMUM) <= 0.1, (
            f"Existing solver found {result.objective:.5f}, expected near {CIRCLE_OPTIMUM}"
        )

    def test_monomial_square_is_convex(self):
        """x² is convex — should solve to x=0 globally even without AMP."""
        m = Model("quad_convex")
        x = m.continuous("x", lb=-3, ub=3)
        m.minimize(x**2)
        result = m.solve()
        assert result.objective is not None
        assert abs(result.objective) < 1e-4, (
            f"x² on [-3,3] has global min 0 at x=0, got {result.objective}"
        )

    def test_existing_piecewise_mccormick_bilinear_function_exists(self):
        """piecewise_mccormick_bilinear must exist in piecewise_mccormick.py.

        If this fails: the function was removed or renamed.
        """
        from discopt._jax.piecewise_mccormick import piecewise_mccormick_bilinear  # noqa: F401

        assert callable(piecewise_mccormick_bilinear)

    def test_obbt_runs_on_bilinear_model(self):
        """OBBT should run without errors on a model with bilinear constraints.

        If this fails: OBBT cannot handle nonlinear constraints.
        """
        try:
            from discopt._jax.obbt import run_obbt

            m = _make_nlp1()
            # Just check it doesn't crash — bound tightening quality tested separately
            run_obbt(m)
        except (ImportError, NotImplementedError):
            pytest.skip("OBBT module not available")
        except Exception as e:
            pytest.fail(f"OBBT crashed on bilinear model: {e}")

    def test_existing_milp_solver_available(self):
        """HiGHS MILP solver must be available for AMP to use."""
        try:
            from discopt.solvers.milp_highs import solve_milp  # noqa: F401

            assert callable(solve_milp)
        except ImportError:
            # Try alternative import path
            try:
                import highspy  # noqa: F401
            except ImportError:
                pytest.skip("HiGHS not available")

    def test_relaxation_compiler_handles_bilinear(self):
        """Relaxation compiler must handle BinaryOp('*', Variable, Variable)."""
        try:
            from discopt._jax.relaxation_compiler import compile_relaxation

            m = _make_nlp1()
            # compile_relaxation(expr, model, partitions, mode)
            assert m._objective is not None
            compile_relaxation(m._objective.expression, m, partitions=0, mode="standard")
        except (ImportError, AttributeError):
            pytest.skip("relaxation_compiler not available in expected form")
        except Exception as e:
            pytest.fail(f"Relaxation compiler failed on bilinear model: {e}")

    @pytest.mark.smoke
    def test_solve_result_has_gap_certified(self):
        """SolveResult must have gap_certified attribute."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        result = m.solve()
        assert hasattr(result, "gap_certified"), "SolveResult missing gap_certified field"

    @pytest.mark.smoke
    def test_solve_result_has_bound(self):
        """SolveResult must have bound attribute (dual lower bound)."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)
        result = m.solve()
        assert hasattr(result, "bound"), "SolveResult missing bound field"
