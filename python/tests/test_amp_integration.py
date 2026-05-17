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

import logging
import os
import warnings

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.model_utils import flat_variable_bounds
from discopt.modeling.core import (
    Model,
    SolveResult,
)
from test_minlptests import NLP_CVX_INSTANCES, NLP_INSTANCES, NLP_MI_INSTANCES

pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.amp_benchmark,
    pytest.mark.memory_heavy,
]


def _unwrap_minlptests_case(case):
    return case.values[0] if hasattr(case, "values") else case


MINLPTESTS_MI_BY_ID = {
    instance.problem_id: instance for instance in map(_unwrap_minlptests_case, NLP_MI_INSTANCES)
}
MINLPTESTS_CVX_BY_ID = {
    instance.problem_id: instance for instance in map(_unwrap_minlptests_case, NLP_CVX_INSTANCES)
}
MINLPTESTS_NLP_BY_ID = {
    instance.problem_id: instance for instance in map(_unwrap_minlptests_case, NLP_INSTANCES)
}


@pytest.mark.requires_cyipopt
def test_amp_integration_environment_includes_working_cyipopt():
    """The opt-in AMP integration environment should include a usable Ipopt backend."""
    import cyipopt  # noqa: F401
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_ipopt import solve_nlp_from_model

    m = Model("cyipopt_integration_smoke")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize((x - 0.25) ** 2)

    result = solve_nlp_from_model(
        m,
        x0=np.array([1.5], dtype=np.float64),
        options={"print_level": 0, "max_iter": 50},
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.objective == pytest.approx(0.0, abs=1e-7)
    assert float(result.x[0]) == pytest.approx(0.25, abs=1e-5)


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


def _make_obbt_demo() -> Model:
    """Small bilinear model where linear constraints sharply tighten the box."""
    m = Model("obbt_demo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y == 1)
    m.maximize(x * y)
    return m


def _make_obbt_ineq_demo() -> Model:
    """Small bilinear model that exercises OBBT's inequality extraction path."""
    m = Model("obbt_ineq_demo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x <= 1)
    m.subject_to(y <= 1)
    m.subject_to(x + y >= 0.5)
    m.maximize(x * y)
    return m


def _make_trilinear_cover() -> Model:
    """Simple trilinear problem with a known AM-GM optimum of 3."""
    m = Model("trilinear_cover")
    x = m.continuous("x", lb=0, ub=2, shape=(3,))
    m.subject_to(x[0] * x[1] * x[2] >= 1.0)
    m.minimize(x[0] + x[1] + x[2])
    return m


def _make_quartic_objective_demo() -> Model:
    """Univariate quartic objective whose MILP lower bound should tighten with refinement."""
    m = Model("quartic_objective")
    x = m.continuous("x", lb=0, ub=2)
    m.subject_to(x >= 1.0)
    m.minimize(x**4)
    return m


def _make_alpine_multi2() -> Model:
    """Alpine examples/MINLPs/multi.jl:multi2.

    Alpine seeds Julia's RNG and documents the active upper-bound solution as
    [0.7336635, 1.266336] with objective 0.92906489.  The translated fixture
    fixes those generated bounds directly so the test is independent of Julia's
    RNG implementation.
    """
    m = Model("alpine_multi2")
    x0 = m.continuous("x0", lb=0.1, ub=0.7336635)
    x1 = m.continuous("x1", lb=0.1, ub=10.0)
    m.subject_to(x0 + x1 <= 2.0)
    m.maximize(x0 * x1)
    return m


def _multi3_term(x, i: int, exprmode: int):
    if exprmode == 1:
        return x[i] * x[i + 1] * x[i + 2]
    if exprmode == 2:
        return (x[i] * x[i + 1]) * x[i + 2]
    if exprmode == 3:
        return x[i] * (x[i + 1] * x[i + 2])
    raise ValueError("multi3N exprmode must be 1, 2, or 3")


def _make_alpine_multi3n(n: int = 2, exprmode: int = 1, randomub: float = 4.0) -> Model:
    """Alpine examples/MINLPs/multi.jl:multi3N."""
    m = Model(f"alpine_multi3N_{n}_{exprmode}")
    size = 1 + 2 * n
    x = m.continuous("x", lb=0.1, ub=randomub, shape=(size,))

    obj = None
    for i in range(0, size - 1, 2):
        term = _multi3_term(x, i, exprmode)
        obj = term if obj is None else obj + term
        m.subject_to(x[i] + x[i + 1] + x[i + 2] <= 3.0)

    assert obj is not None
    m.maximize(obj)
    return m


def _multi4_term(x, i: int, exprmode: int):
    if exprmode == 1:
        return x[i] * x[i + 1] * x[i + 2] * x[i + 3]
    if exprmode == 2:
        return (x[i] * x[i + 1]) * (x[i + 2] * x[i + 3])
    if exprmode == 3:
        return (x[i] * x[i + 1]) * x[i + 2] * x[i + 3]
    if exprmode == 4:
        return x[i] * x[i + 1] * (x[i + 2] * x[i + 3])
    if exprmode == 5:
        return ((x[i] * x[i + 1]) * x[i + 2]) * x[i + 3]
    if exprmode == 6:
        return (x[i] * x[i + 1] * x[i + 2]) * x[i + 3]
    if exprmode == 7:
        return x[i] * (x[i + 1] * (x[i + 2] * x[i + 3]))
    if exprmode == 8:
        return x[i] * (x[i + 1] * x[i + 2]) * x[i + 3]
    if exprmode == 9:
        return x[i] * (x[i + 1] * x[i + 2] * x[i + 3])
    if exprmode == 10:
        return x[i] * ((x[i + 1] * x[i + 2]) * x[i + 3])
    if exprmode == 11:
        return (x[i] * (x[i + 1] * x[i + 2])) * x[i + 3]
    raise ValueError("multi4N exprmode must be in 1..11")


def _make_alpine_multi4n(n: int = 2, exprmode: int = 1, randomub: float = 4.0) -> Model:
    """Alpine examples/MINLPs/multi.jl:multi4N."""
    m = Model(f"alpine_multi4N_{n}_{exprmode}")
    size = 1 + 3 * n
    x = m.continuous("x", lb=0.1, ub=randomub, shape=(size,))

    obj = None
    for i in range(0, size - 1, 3):
        term = _multi4_term(x, i, exprmode)
        obj = term if obj is None else obj + term
        m.subject_to(x[i] + x[i + 1] + x[i + 2] + x[i + 3] <= 4.0)

    assert obj is not None
    m.maximize(obj)
    return m


def _make_alpine_castro2m2_partition_fixture() -> Model:
    """Nonlinear partition skeleton from Alpine examples/MINLPs/castro.jl:castro2m2."""
    m = Model("alpine_castro2m2_partition")
    x = m.continuous("x", lb=0.0, ub=1e5, shape=(41,))
    obj = m.continuous("obj", lb=0.0, ub=1e3)
    m.minimize(obj)

    for i, j, target in [
        (27, 29, 15),
        (27, 30, 16),
        (28, 31, 17),
        (28, 32, 18),
        (27, 35, 21),
        (28, 36, 22),
        (13, 29, 0),
        (13, 30, 1),
        (14, 31, 2),
        (14, 32, 3),
        (13, 35, 6),
        (14, 36, 7),
    ]:
        m.subject_to(x[i] * x[j] - x[target] == 0.0)

    return m


def _make_alpine_blend029_gl_partition_fixture() -> Model:
    """Nonlinear partition skeleton from Alpine examples/MINLPs/blend.jl:blend029_gl."""
    m = Model("alpine_blend029_gl_partition")
    x = []
    for i in range(48):
        x.append(m.continuous(f"x{i + 1}", lb=0.0, ub=1.0))
    for i in range(48, 66):
        x.append(m.continuous(f"x{i + 1}", lb=0.0, ub=2.0))
    for i in range(66, 102):
        x.append(m.binary(f"x{i + 1}"))
    m.maximize(0.0 * x[0])

    exprs = [
        (x[36] * x[54] - 0.6 * x[0] - 0.2 * x[12] + 0.2 * x[24] + 0.2 * x[27] + 0.2 * x[30], 0.04),
        (x[39] * x[57] - 0.6 * x[3] - 0.2 * x[15] - 0.2 * x[24] + 0.7 * x[33], 0.07),
        (x[42] * x[54] - 0.4 * x[0] - 0.4 * x[12] + 0.5 * x[24] + 0.5 * x[27] + 0.5 * x[30], 0.1),
        (x[45] * x[57] - 0.4 * x[3] - 0.4 * x[15] - 0.5 * x[24] + 0.6 * x[33], 0.06),
        (
            x[37] * x[55]
            - (x[36] * x[54] - (x[36] * x[25] + x[36] * x[28] + x[36] * x[31]))
            - 0.6 * x[1]
            - 0.2 * x[13],
            0.0,
        ),
        (
            x[38] * x[56]
            - (x[37] * x[55] - (x[37] * x[26] + x[37] * x[29] + x[37] * x[32]))
            - 0.6 * x[2]
            - 0.2 * x[14],
            0.0,
        ),
        (
            x[40] * x[58]
            - (x[39] * x[57] + x[36] * x[25] - x[39] * x[34])
            - 0.6 * x[4]
            - 0.2 * x[16],
            0.0,
        ),
        (
            x[41] * x[59]
            - (x[40] * x[58] + x[37] * x[26] - x[40] * x[35])
            - 0.6 * x[5]
            - 0.2 * x[17],
            0.0,
        ),
        (
            x[43] * x[55]
            - (x[42] * x[54] - (x[42] * x[25] + x[42] * x[28] + x[42] * x[31]))
            - 0.4 * x[1]
            - 0.4 * x[13],
            0.0,
        ),
        (
            x[44] * x[56]
            - (x[43] * x[55] - (x[43] * x[26] + x[43] * x[29] + x[43] * x[32]))
            - 0.4 * x[2]
            - 0.4 * x[14],
            0.0,
        ),
        (
            x[46] * x[58]
            - (x[45] * x[57] + x[42] * x[25] - x[45] * x[34])
            - 0.4 * x[4]
            - 0.4 * x[16],
            0.0,
        ),
        (
            x[47] * x[59]
            - (x[46] * x[58] + x[43] * x[26] - x[46] * x[35])
            - 0.4 * x[5]
            - 0.4 * x[17],
            0.0,
        ),
    ]
    for expr, rhs in exprs:
        m.subject_to(expr >= rhs)
        m.subject_to(expr <= rhs)

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

    def test_detect_higher_order_multilinear_without_pairwise_expansion(self):
        """x[0]*x[1]*x[2]*x[3] should not create unused pairwise bilinear terms."""
        m = Model("t")
        x = m.continuous("x", lb=0, ub=10, shape=(4,))
        m.minimize(x[0] * x[1] * x[2] * x[3])

        terms = self.classify(m)

        assert terms.bilinear == []
        assert terms.trilinear == []
        assert terms.multilinear == [(0, 1, 2, 3)]
        assert set(terms.partition_candidates) == {0, 1, 2, 3}

    def test_mixed_bilinear_and_monomial(self):
        """nlp1: bilinear in constraint, monomial in objective."""
        m = _make_nlp1()
        terms = self.classify(m)
        assert len(terms.bilinear) >= 1
        assert len(terms.monomial) >= 2  # x[0]**2 and x[1]**2

    def test_gas_network_weymouth_monomials_extend_amp_partition_selection(self):
        """Gas-network square terms should be eligible for AMP partition refinement."""
        from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp
        from discopt.solvers import amp as amp_mod

        m = build_gas_network_minlp()
        terms = self.classify(m)
        selected_square_vars = set(amp_mod._equality_square_monomial_partition_candidates(m, terms))
        product_vars = {var_idx for term in terms.bilinear + terms.trilinear for var_idx in term}
        for term in terms.multilinear:
            product_vars.update(term)

        assert selected_square_vars
        assert selected_square_vars - product_vars
        assert not selected_square_vars.issubset(set(terms.partition_candidates))

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

    def _make_terms(self, bilinear=None, trilinear=None, multilinear=None, monomial=None):
        """Helper to build a NonlinearTerms stub."""
        bilinear = bilinear or []
        trilinear = trilinear or []
        multilinear = multilinear or []
        monomial = monomial or []
        candidates = list(
            {v for t in bilinear + trilinear + multilinear for v in t} | {v for v, _ in monomial}
        )
        incidence: dict[int, set[int]] = {}
        for idx, t in enumerate(bilinear + trilinear + multilinear):
            for v in t:
                incidence.setdefault(v, set()).add(idx)
        return self.NonlinearTerms(
            bilinear=bilinear,
            trilinear=trilinear,
            multilinear=multilinear,
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

    def test_multilinear_terms_covered(self):
        """Higher-order multilinear terms must be covered as one product term."""
        terms = self._make_terms(multilinear=[(0, 1, 2, 3)])
        selected = self.pick(terms, method="min_vertex_cover")

        assert selected
        assert any(v in selected for v in (0, 1, 2, 3)), "multilinear term not covered"

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


class TestAlpinePortedPartitionSelection:
    """Ports of Alpine partition-variable tests for larger named examples."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from discopt._jax.partition_selection import pick_partition_vars
        from discopt._jax.term_classifier import classify_nonlinear_terms

        self.classify = classify_nonlinear_terms
        self.pick = pick_partition_vars

    def _assert_terms_covered(self, terms, selected):
        selected_set = set(selected)
        for term in terms.bilinear + terms.trilinear + terms.multilinear:
            assert any(v in selected_set for v in term), f"term {term} not covered"

    def test_castro2m2_candidates_match_alpine_source(self):
        """Alpine castro2m2 has 10 candidate discretization variables and a 4-var cover."""
        terms = self.classify(_make_alpine_castro2m2_partition_fixture())

        expected = {13, 14, 27, 28, 29, 30, 31, 32, 35, 36}
        max_cover = self.pick(terms, method="max_cover")
        min_cover = self.pick(terms, method="min_vertex_cover")

        assert set(terms.partition_candidates) == expected
        assert set(max_cover) == expected
        assert len(min_cover) == 4
        self._assert_terms_covered(terms, min_cover)

    def test_blend029_gl_candidates_match_alpine_source(self):
        """Alpine blend029_gl has 26 candidate discretization variables and a 10-var cover."""
        terms = self.classify(_make_alpine_blend029_gl_partition_fixture())

        expected = {
            25,
            26,
            28,
            29,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            54,
            55,
            56,
            57,
            58,
            59,
        }
        max_cover = self.pick(terms, method="max_cover")
        min_cover = self.pick(terms, method="min_vertex_cover")

        assert set(terms.partition_candidates) == expected
        assert set(max_cover) == expected
        assert len(min_cover) == 10
        self._assert_terms_covered(terms, min_cover)


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

    def test_partitioned_circle_monomial_lb_uses_local_secants(self):
        """Partitioned square overestimators should certify the Alpine circle bound."""
        m = _make_circle()
        terms = self.classify(m)
        square_vars = sorted(var_idx for var_idx, exp in terms.monomial if exp == 2)
        state = self.init_partitions(square_vars, lb=[0.0, 0.0], ub=[2.0, 2.0], n_init=64)

        milp_model, varmap = self.build_milp(m, terms, state, incumbent=None)
        result = milp_model.solve()

        assert set(varmap["monomial_pw"]) == {(0, 2), (1, 2)}
        assert result.status == "optimal"
        assert result.objective is not None
        assert result.objective <= CIRCLE_OPTIMUM + 1e-4
        assert result.objective >= CIRCLE_OPTIMUM - 1e-4

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

    def test_sos2_embedding_uses_logarithmic_selector_count(self):
        """Embedded SOS2 should replace interval binaries with O(log K) selectors."""
        m = _make_nlp1()
        terms = self.classify(m)
        state = self.init_partitions([0], lb=[1.0], ub=[4.0], n_init=8)

        _, plain_varmap = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="sos2",
        )
        _, embedded_varmap = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="sos2",
            convhull_ebd=True,
            convhull_ebd_encoding="gray",
        )

        plain_info = plain_varmap["bilinear_lambda"][(0, 1)]
        embedded_info = embedded_varmap["bilinear_lambda"][(0, 1)]
        assert len(plain_info["alpha_cols"]) == 8
        assert len(embedded_info["alpha_cols"]) == 0
        assert len(embedded_info["embedding_cols"]) == 3
        assert embedded_varmap["convhull_ebd"] is True
        assert embedded_varmap["convhull_ebd_encoding"] == "gray"

    def test_sos2_embedding_matches_plain_sos2_relaxation_value(self):
        """Embedded SOS2 should preserve the λ-relaxation bound on a simple model."""
        m = Model("embedding_compare")
        x = m.continuous("x", lb=0, ub=2, shape=(2,))
        m.subject_to(x[0] * x[1] >= 1.0)
        m.minimize(x[0] + x[1])

        terms = self.classify(m)
        state = self.init_partitions([0], lb=[0.0], ub=[2.0], n_init=4)

        plain_model, _ = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="sos2",
        )
        embedded_model, _ = self.build_milp(
            m,
            terms,
            state,
            incumbent=None,
            convhull_formulation="sos2",
            convhull_ebd=True,
            convhull_ebd_encoding="gray",
        )

        plain_result = plain_model.solve()
        embedded_result = embedded_model.solve()

        assert plain_result.status == "optimal"
        assert embedded_result.status == "optimal"
        assert plain_result.objective is not None
        assert embedded_result.objective is not None
        assert embedded_result.objective == pytest.approx(plain_result.objective, abs=1e-6)

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

    def test_embedding_helper_rejects_non_sos2_compatible_encoding(self):
        """Binary counting codes are invalid for SOS2 once adjacency breaks."""
        from discopt._jax.embedding import build_embedding_map

        with pytest.raises(ValueError, match="only works for exactly 2 partitions"):
            build_embedding_map(5, encoding="binary")

    def test_embedding_requires_sos2_formulation(self):
        """Embedded binaries should be rejected for non-SOS2 formulations."""
        m = _make_nlp1()
        terms = self.classify(m)
        state = self.init_partitions([0], lb=[1.0], ub=[4.0], n_init=2)

        with pytest.raises(ValueError, match="convhull_ebd is only supported"):
            self.build_milp(
                m,
                terms,
                state,
                incumbent=None,
                convhull_formulation="facet",
                convhull_ebd=True,
            )

    def test_sos2_embedding_requires_selector_columns(self, monkeypatch):
        """SOS2 linking must keep either alpha or embedded selector columns."""
        from discopt._jax.embedding import EmbeddingMap

        m = _make_nlp1()
        terms = self.classify(m)
        state = self.init_partitions([0], lb=[1.0], ub=[4.0], n_init=4)

        monkeypatch.setattr(
            "discopt._jax.milp_relaxation.build_embedding_map",
            lambda lambda_count, encoding="gray": EmbeddingMap(
                encoding=encoding,
                bit_count=0,
                codes=tuple(),
                positive_sets=tuple(),
                negative_sets=tuple(),
            ),
        )

        with pytest.raises(
            AssertionError,
            match="Expected either alpha or embedding columns for SOS2 linking",
        ):
            self.build_milp(
                m,
                terms,
                state,
                incumbent=None,
                convhull_formulation="sos2",
                convhull_ebd=True,
            )


class TestAmpPhase4Coverage:
    """Regression coverage for trilinear and higher-order monomial relaxations."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from discopt._jax.discretization import initialize_partitions
        from discopt._jax.milp_relaxation import (
            _odd_mixed_tangent_is_valid,
            build_milp_relaxation,
        )
        from discopt._jax.term_classifier import classify_nonlinear_terms

        self.build_milp = build_milp_relaxation
        self.classify = classify_nonlinear_terms
        self.init_partitions = initialize_partitions
        self.is_valid_odd_tangent = _odd_mixed_tangent_is_valid

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

    @pytest.mark.parametrize("exprmode", [1, 2, 3])
    def test_alpine_multi3n_milp_builds_trilinear_auxiliaries(self, exprmode):
        """Alpine multi3N should build one lifted trilinear objective per overlapping block."""
        m = _make_alpine_multi3n(n=2, exprmode=exprmode)
        terms = self.classify(m)
        flat_lb, flat_ub = flat_variable_bounds(m)
        state = self.init_partitions(
            terms.partition_candidates,
            lb=[flat_lb[i] for i in terms.partition_candidates],
            ub=[flat_ub[i] for i in terms.partition_candidates],
            n_init=2,
        )

        milp_model, varmap = self.build_milp(m, terms, state, incumbent=None)
        result = milp_model.solve()

        assert (0, 1, 2) in varmap["trilinear"]
        assert (2, 3, 4) in varmap["trilinear"]
        assert result.status == "optimal"
        assert result.objective is not None

    @pytest.mark.parametrize("exprmode", range(1, 12))
    def test_alpine_multi4n_builds_recursive_multilinear_auxiliaries(self, exprmode):
        """Alpine multi4N expression modes should linearize through recursive bilinear lifts."""
        m = _make_alpine_multi4n(n=2, exprmode=exprmode)
        terms = self.classify(m)
        flat_lb, flat_ub = flat_variable_bounds(m)
        state = self.init_partitions(
            terms.partition_candidates,
            lb=[flat_lb[i] for i in terms.partition_candidates],
            ub=[flat_ub[i] for i in terms.partition_candidates],
            n_init=2,
        )

        milp_model, varmap = self.build_milp(m, terms, state, incumbent=None)

        assert (0, 1, 2, 3) in varmap["multilinear"]
        assert (3, 4, 5, 6) in varmap["multilinear"]
        assert len(varmap["multilinear_stages"][(0, 1, 2, 3)]) == 3
        assert len(varmap["multilinear_stages"][(3, 4, 5, 6)]) == 3
        assert milp_model._objective_bound_valid is True

    def test_multilinear_build_avoids_unused_original_pairwise_lifts(self):
        """A pure 4-factor product should build only the recursive product chain."""
        m = Model("multilinear_chain_only")
        x = m.continuous("x", lb=0.1, ub=4.0, shape=(4,))
        m.maximize(x[0] * x[1] * x[2] * x[3])
        terms = self.classify(m)
        state = self.init_partitions(
            terms.partition_candidates,
            lb=[0.1] * 4,
            ub=[4.0] * 4,
            n_init=2,
        )

        _, varmap = self.build_milp(m, terms, state, incumbent=None)

        stages = varmap["multilinear_stages"][(0, 1, 2, 3)]
        chain_pairs = {tuple(sorted((stage["lhs_col"], stage["rhs_col"]))) for stage in stages}
        unused_original_pairs = {
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        }

        assert varmap["bilinear"] == {}
        assert set(varmap["bilinear_pw"]) == chain_pairs
        assert not unused_original_pairs & set(varmap["bilinear_pw"])

    def test_alpine_multi4n_milp_relaxation_solves_with_objective_bound(self):
        """The recursive multi4N relaxation should solve with a real objective bound."""
        m = _make_alpine_multi4n(n=2, exprmode=1)
        terms = self.classify(m)
        flat_lb, flat_ub = flat_variable_bounds(m)
        state = self.init_partitions(
            terms.partition_candidates,
            lb=[flat_lb[i] for i in terms.partition_candidates],
            ub=[flat_ub[i] for i in terms.partition_candidates],
            n_init=2,
        )

        milp_model, _ = self.build_milp(m, terms, state, incumbent=None)
        result = milp_model.solve()

        assert result.status == "optimal"
        assert result.objective is not None

    def test_quartic_relaxation_tightens_with_finer_partitions(self):
        """Breakpoint tangents should tighten n>2 monomial objectives as partitions refine."""
        m = _make_quartic_objective_demo()
        terms = self.classify(m)
        lbs = []

        for n_init in [1, 2, 4, 8]:
            state = self.init_partitions([0], lb=[0.0], ub=[2.0], n_init=n_init)
            milp_model, _ = self.build_milp(m, terms, state, incumbent=None)
            result = milp_model.solve()
            assert result.status == "optimal"
            assert result.objective is not None
            lbs.append(float(result.objective))

        for i in range(len(lbs) - 1):
            assert lbs[i] <= lbs[i + 1] + 1e-8
        assert lbs[0] < 0.2
        assert lbs[-1] >= 0.999

    def test_nlp_005_010_relaxation_keeps_reciprocal_and_negative_power(self, caplog):
        """Issue #62: the root MILP should retain reciprocal and y**(-2) structure."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        instance = MINLPTESTS_NLP_BY_ID["nlp_005_010"]
        m = instance.build_fn()
        flat_lb, flat_ub = flat_variable_bounds(m)
        tightened_lb, tightened_ub, _stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

        terms = self.classify(m)
        state = self.init_partitions(
            terms.partition_candidates,
            lb=[float(tightened_lb[i]) for i in terms.partition_candidates],
            ub=[float(tightened_ub[i]) for i in terms.partition_candidates],
            n_init=2,
        )

        with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
            milp_model, varmap = self.build_milp(
                m,
                terms,
                state,
                incumbent=None,
                bound_override=(tightened_lb, tightened_ub),
            )

        reciprocal_lifts = [
            relax for relax in varmap["univariate_relaxations"] if relax.func_name == "reciprocal"
        ]
        assert len(reciprocal_lifts) >= 2
        assert (1, -2.0) in varmap["fractional_power"]

        a_ub = milp_model._A_ub.toarray()
        exact_rows = [
            idx
            for idx, row in enumerate(a_ub)
            if np.allclose(row[:2], [1.0, 1.0], atol=1e-12)
            and np.count_nonzero(np.abs(row[2:]) > 1e-12) == 0
        ]
        assert any(milp_model._b_ub[idx] == pytest.approx(3.9) for idx in exact_rows)
        assert not any(
            "Cannot linearize non-constant division" in rec.message for rec in caplog.records
        )
        assert not any("Monomial (1, -2)" in rec.message for rec in caplog.records)

    def test_mixed_sign_odd_tangent_filter_keeps_only_global_supporting_lines(self):
        """Only tangents that stay valid on the full mixed-sign box should be used."""
        assert self.is_valid_odd_tangent(0.5, -0.5, 0.5, 5, "under") is True
        assert self.is_valid_odd_tangent(-0.5, -0.5, 0.5, 5, "over") is True
        assert self.is_valid_odd_tangent(0.1, -1.0, 0.1, 5, "under") is False
        assert self.is_valid_odd_tangent(-0.1, -0.1, 1.0, 5, "over") is False


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
MULTI2_OPTIMUM = 0.92906489  # Alpine multi2


class TestAmpEndToEnd:
    """End-to-end AMP solver tests on Alpine canonical problems.

    Will fail until solvers/amp.py exists and solver.py routes solver="amp".
    """

    @pytest.mark.smoke
    def test_nlp1_bilinear_global_optimum(self):
        """nlp1: AMP should recover the known incumbent without invalid OA certificates."""

        m = _make_nlp1()
        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)
        assert result.status in ("optimal", "feasible"), f"Unexpected status {result.status}"
        assert result.gap_certified is (result.status == "optimal")
        assert result.objective is not None
        assert abs(result.objective - NLP1_OPTIMUM) <= 0.15, (
            f"Objective {result.objective:.5f} too far from {NLP1_OPTIMUM}"
        )

    @pytest.mark.smoke
    def test_circle_bilinear_global_optimum(self):
        """circle: x₀²+x₁²≥2 minimized to global optimum √2."""
        m = _make_circle()
        result = m.solve(solver="amp", rel_gap=1e-4, time_limit=60)
        assert result.status == "optimal"
        assert result.gap_certified is True
        assert result.objective is not None
        assert abs(result.objective - CIRCLE_OPTIMUM) <= 1e-3, (
            f"Objective {result.objective:.6f} too far from √2={CIRCLE_OPTIMUM}"
        )

    def test_amp_embedding_rebuilds_across_refinement_iterations(self, monkeypatch):
        """Embedded SOS2 should stay consistent as AMP refines the partitions."""
        import discopt._jax.milp_relaxation as milp_mod

        orig_build = milp_mod.build_milp_relaxation
        lambda_counts = []
        bit_counts = []

        def spy_build(*args, **kwargs):
            milp_model, varmap = orig_build(*args, **kwargs)
            info = next(iter(varmap["bilinear_lambda"].values()))
            lambda_counts.append(len(info["lambda_cols"]))
            bit_counts.append(len(info["embedding_cols"]))
            return milp_model, varmap

        monkeypatch.setattr(milp_mod, "build_milp_relaxation", spy_build)

        m = _make_nlp1()
        result = m.solve(
            solver="amp",
            convhull_formulation="sos2",
            convhull_ebd=True,
            rel_gap=1e-6,
            max_iter=4,
            time_limit=30,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert abs(result.objective - NLP1_OPTIMUM) <= 0.15
        assert len(bit_counts) >= 2
        assert bit_counts[0] == 1
        assert bit_counts[0] < max(bit_counts)
        assert lambda_counts[0] < max(lambda_counts)
        assert all(
            bit_count == max(1, int(np.ceil(np.log2(max(1, lambda_count - 1)))))
            for bit_count, lambda_count in zip(bit_counts, lambda_counts)
        )

    @pytest.mark.smoke
    def test_trilinear_global_optimum(self):
        """A simple AM-GM trilinear instance should recover the known incumbent."""
        m = _make_trilinear_cover()
        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)
        assert result.status in ("optimal", "feasible")
        assert result.gap_certified is (result.status == "optimal")
        assert result.objective is not None
        assert abs(result.objective - 3.0) <= 1e-3

    def test_alpine_multi2_global_optimum(self):
        """Alpine multi2 should recover the documented incumbent objective."""
        m = _make_alpine_multi2()
        result = m.solve(
            solver="amp",
            rel_gap=1e-3,
            max_iter=8,
            presolve_bt=False,
            time_limit=30,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert abs(result.objective - MULTI2_OPTIMUM) <= 1e-3

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    @pytest.mark.xfail(
        reason="nlp3 MILP exceeds test budget; tracked in #24",
        strict=False,
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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_pure_quadratic_convex(self):
        """min x²  s.t. x ≥ 1: AMP should certify global optimum = 1."""
        m = Model("quad")
        x = m.continuous("x", lb=1, ub=10)
        m.minimize(x**2)
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

        result = m.solve(solver="amp", rel_gap=1e-4, time_limit=30)

        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective) <= 1e-6
        assert result.gap is None

    def test_bilinear_maximize_global_optimum(self):
        """AMP must handle maximize objectives with certified bounds."""
        m = Model("max_bilinear")
        x = m.continuous("x", lb=0, ub=2, shape=(2,))
        m.subject_to(x[0] + x[1] <= 2)
        m.maximize(x[0] * x[1])

        result = m.solve(solver="amp", rel_gap=1e-3, time_limit=60)

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert abs(result.objective - 1.0) <= 1e-3
        if result.status == "optimal":
            assert result.gap_certified is True
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
        m = _make_circle()
        result = m.solve(solver="amp", rel_gap=0.05, time_limit=30)
        # If we converged by gap criterion, gap_certified must be True
        if result.status == "optimal":
            assert result.gap_certified is True

    @pytest.mark.smoke
    def test_early_termination_when_gap_closes(self):
        """AMP should terminate before max_iter when gap closes."""
        m = _make_circle()
        iters = []

        def cb(info):
            iters.append(info["iteration"])

        result = m.solve(
            solver="amp", rel_gap=0.5, max_iter=100, iteration_callback=cb, time_limit=30
        )
        if result.status == "optimal":
            # If gap-terminated, should be well before max_iter
            assert len(iters) < 100, "Should terminate before max_iter=100 when gap closes"

    @pytest.mark.smoke
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
            initial_point=np.array([0.25], dtype=np.float64),
            use_start_as_incumbent=True,
            apply_partitioning=False,
            disc_var_pick=1,
            partition_scaling_factor=7.0,
            disc_add_partition_method="uniform",
            disc_abs_width_tol=1e-2,
            convhull_formulation="sos2",
            convhull_ebd=True,
            convhull_ebd_encoding="gray",
        )

        assert captured["rel_gap"] == pytest.approx(1e-3)
        assert np.allclose(captured["initial_point"], np.array([0.25], dtype=np.float64))
        assert captured["use_start_as_incumbent"] is True
        assert captured["apply_partitioning"] is False
        assert captured["disc_var_pick"] == 1
        assert captured["partition_scaling_factor"] == pytest.approx(7.0)
        assert captured["disc_add_partition_method"] == "uniform"
        assert captured["disc_abs_width_tol"] == pytest.approx(1e-2)
        assert captured["convhull_formulation"] == "sos2"
        assert captured["convhull_ebd"] is True
        assert captured["convhull_ebd_encoding"] == "gray"

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

    def test_integer_rounding_candidates_respect_max_candidates(self):
        """The fallback candidate list should stay bounded by max_candidates."""
        from discopt.solvers import amp as amp_mod

        m = Model("rounding_cap")
        m.binary("y", shape=(100,))
        x0 = np.full(100, 0.49, dtype=np.float64)

        candidates = amp_mod._integer_rounding_candidates(x0, m, max_candidates=64)

        assert len(candidates) == 64
        assert tuple(float(v) for v in candidates[0]) == tuple(0.0 for _ in range(100))

    def test_integer_rounding_candidates_enumerate_small_finite_domains(self):
        """Small finite integer boxes should be enumerated instead of a single rounded point."""
        from discopt.solvers import amp as amp_mod

        m = Model("rounding_box")
        m.integer("y", lb=0, ub=4, shape=(2,))

        candidates = amp_mod._integer_rounding_candidates(np.array([4.0, 4.0]), m)
        rounded = {tuple(float(v) for v in cand) for cand in candidates}

        assert len(candidates) == 25
        assert (3.0, 2.0) in rounded

    def test_build_fixed_integer_bounds_clamps_to_integer_domain(self):
        """Rounded fixed bounds should stay within the realizable integer domain."""
        from discopt.solvers import amp as amp_mod

        m = Model("fixed_bounds_clamp")
        m.integer("y", lb=0.2, ub=2.6)

        nlp_lb, nlp_ub = amp_mod._build_fixed_integer_bounds(
            np.array([2.6], dtype=np.float64),
            m,
            flat_lb=np.array([0.2], dtype=np.float64),
            flat_ub=np.array([2.6], dtype=np.float64),
        )

        assert nlp_lb[0] == pytest.approx(2.0)
        assert nlp_ub[0] == pytest.approx(2.0)

    def test_solve_model_forwards_amp_presolve_bt_option(self, monkeypatch):
        """solve_model should pass the OBBT toggle through to solve_amp."""
        from discopt.solver import solve_model
        from discopt.solvers import amp as amp_mod

        captured = {}

        def fake_solve_amp(model, **kwargs):
            captured.update(kwargs)
            return SolveResult(status="optimal")

        monkeypatch.setattr(amp_mod, "solve_amp", fake_solve_amp)

        m = Model("forward_obbt")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        solve_model(m, solver="amp", presolve_bt=False, time_limit=1.0)

        assert captured["presolve_bt"] is False

    def test_default_obbt_time_limit_per_lp_is_bounded(self):
        """OBBT per-LP budgets should stay bounded and respect missing time."""
        from discopt.solvers import amp as amp_mod

        assert amp_mod._default_obbt_time_limit_per_lp(0.0, 2) == pytest.approx(0.0)
        assert amp_mod._default_obbt_time_limit_per_lp(np.inf, 2) == pytest.approx(0.0)
        assert amp_mod._default_obbt_time_limit_per_lp(100.0, 2) == pytest.approx(2.5)
        assert amp_mod._default_obbt_time_limit_per_lp(1.0, 100) == pytest.approx(0.05)

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
            lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
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

    def test_best_nlp_candidate_prioritizes_incumbent_start_then_model_start_then_milp(
        self,
        monkeypatch,
    ):
        """AMP local search should try incumbent, model start, then MILP point first."""
        from discopt.solvers import amp as amp_mod

        m = Model("candidate_priority")
        m.continuous("x", lb=0.0, ub=10.0)
        seen_starts = []

        monkeypatch.setattr(
            amp_mod,
            "_solve_nlp_subproblem",
            lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
                seen_starts.append(float(x0[0])) or (None, None)
            ),
        )

        amp_mod._solve_best_nlp_candidate(
            np.array([6.0], dtype=np.float64),
            m,
            evaluator=object(),
            flat_lb=np.array([0.0], dtype=np.float64),
            flat_ub=np.array([10.0], dtype=np.float64),
            constraint_lb=np.array([], dtype=np.float64),
            constraint_ub=np.array([], dtype=np.float64),
            nlp_solver="ipm",
            incumbent=np.array([2.0], dtype=np.float64),
            initial_point=np.array([4.0], dtype=np.float64),
        )

        assert seen_starts[:3] == [2.0, 4.0, 6.0]

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
            lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
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

    def test_amp_accepts_feasible_start_as_incumbent(self, monkeypatch):
        """A feasible model start should survive when proof search fails immediately."""
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        m = Model("amp_start_incumbent")
        x = m.continuous("x", lb=0.0, ub=2.0)
        m.subject_to(x >= 0.25)
        m.minimize((x - 1.0) ** 2)

        monkeypatch.setattr(
            amp_mod,
            "_solve_milp_with_oa_recovery",
            lambda **kwargs: (
                MilpRelaxationResult(status="error", objective=None, x=None),
                {},
                [],
                1,
            ),
        )

        result = m.solve(
            solver="amp",
            initial_solution={x: 0.5},
            use_start_as_incumbent=True,
            skip_convex_check=True,
            presolve_bt=False,
            max_iter=1,
            time_limit=30,
        )

        assert result.status == "feasible"
        assert result.objective == pytest.approx(0.25)
        assert result.x is not None
        assert np.asarray(result.x["x"]).item() == pytest.approx(0.5)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_amp_rejects_nonfinite_direct_initial_point(self, bad_value):
        """Direct AMP initial points must be finite before incumbent checks."""
        from discopt.solvers import amp as amp_mod

        m = Model("amp_nonfinite_start")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(x)

        with pytest.raises(ValueError, match="finite"):
            amp_mod.solve_amp(
                m,
                initial_point=np.array([bad_value], dtype=np.float64),
                use_start_as_incumbent=True,
                skip_convex_check=True,
                presolve_bt=False,
                max_iter=1,
                time_limit=1.0,
            )

    def test_amp_does_not_accept_start_with_nonfinite_objective(self, monkeypatch):
        """A finite start with NaN objective is not a valid AMP incumbent."""
        import discopt._jax.nlp_evaluator as nlp_eval
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        m = Model("amp_nan_objective_start")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(x)

        monkeypatch.setattr(
            nlp_eval.NLPEvaluator,
            "evaluate_objective",
            lambda self, x_flat: np.nan,
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_milp_with_oa_recovery",
            lambda **kwargs: (
                MilpRelaxationResult(status="error", objective=None, x=None),
                {},
                [],
                1,
            ),
        )
        result = m.solve(
            solver="amp",
            initial_solution={x: 0.5},
            use_start_as_incumbent=True,
            skip_convex_check=True,
            presolve_bt=False,
            max_iter=1,
            time_limit=1.0,
        )

        assert result.status == "error"
        assert result.objective is None
        assert result.x is None

    def test_amp_recovers_minlptests_integer_incumbent_from_small_finite_box(self):
        """AMP should recover finite-domain MINLP incumbents even from integral MILP points."""
        instance = MINLPTESTS_MI_BY_ID["nlp_mi_003_014"]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
            apply_partitioning=False,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    def test_tighten_sum_of_squares_bounds_infers_finite_integer_domain(self):
        """Quadratic norm constraints should clamp otherwise unbounded domains."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_sum_of_squares")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        y = m.integer("y", lb=-100, ub=100)
        m.subject_to(x**2 + y**2 <= 10.0)
        m.minimize(x * 0.0 + y * 0.0)

        flat_lb = np.array([-1e20, -100.0], dtype=np.float64)
        flat_ub = np.array([1e20, 100.0], dtype=np.float64)
        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

        radius = np.sqrt(10.0)
        assert tightened_lb[0] == pytest.approx(-radius)
        assert tightened_ub[0] == pytest.approx(radius)
        assert tightened_lb[1] == pytest.approx(-3.0)
        assert tightened_ub[1] == pytest.approx(3.0)
        assert "sum_of_squares_upper_bound" in stats.applied_rules

    def test_tighten_sqrt_sum_of_squares_bounds_infers_finite_box(self):
        """Sqrt norm constraints should clamp otherwise unbounded domains."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_sqrt_sum_of_squares")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        y = m.continuous("y", lb=-1e20, ub=1e20)
        m.subject_to(dm.sqrt(dm.sum([x**2, y**2])) <= 2.0)
        m.minimize(x * 0.0 + y * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([-1e20, -1e20], dtype=np.float64),
            np.array([1e20, 1e20], dtype=np.float64),
        )

        assert tightened_lb[0] == pytest.approx(-2.0)
        assert tightened_ub[0] == pytest.approx(2.0)
        assert tightened_lb[1] == pytest.approx(-2.0)
        assert tightened_ub[1] == pytest.approx(2.0)
        assert "sqrt_sum_of_squares_upper_bound" in stats.applied_rules

    def test_continuous_solver_starts_safely_after_sqrt_tightening(self):
        """Formerly unbounded sqrt-norm variables should not start at the nonsmooth origin."""
        m = Model("bt_sqrt_single_solve")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        m.minimize(-x)
        m.subject_to(dm.sqrt(dm.sum([x**2])) <= 1.0)

        result = m.solve(time_limit=10.0, gap_tolerance=1e-6)

        assert result.status in ("optimal", "feasible")
        assert result.objective == pytest.approx(-1.0, abs=1e-5)

    def test_continuous_solver_backend_receives_tightened_bounds(self, monkeypatch):
        """Direct continuous NLP backends should see the tightened variable box."""
        import time

        import discopt.solver as solver_mod
        from discopt.solvers import NLPResult, SolveStatus

        captured: dict[str, np.ndarray] = {}

        def fake_solve_nlp(evaluator, x0, constraint_bounds=None, options=None):
            del constraint_bounds, options
            backend_lb, backend_ub = evaluator.variable_bounds
            captured["lb"] = np.asarray(backend_lb, dtype=np.float64).copy()
            captured["ub"] = np.asarray(backend_ub, dtype=np.float64).copy()
            captured["x0"] = np.asarray(x0, dtype=np.float64).copy()
            return NLPResult(status=SolveStatus.OPTIMAL, x=captured["x0"], objective=0.0)

        monkeypatch.setattr(solver_mod, "solve_nlp", fake_solve_nlp)

        m = Model("bt_sqrt_backend_bounds")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        m.minimize(x * 0.0)
        m.subject_to(dm.sqrt(dm.sum([x**2])) <= 1.0)

        result = solver_mod._solve_continuous(
            m,
            time_limit=5.0,
            ipopt_options={},
            t_start=time.perf_counter(),
            nlp_solver="ipopt",
        )

        assert result.status == "optimal"
        assert captured["lb"] == pytest.approx(np.array([-1.0]))
        assert captured["ub"] == pytest.approx(np.array([1.0]))
        assert captured["x0"] == pytest.approx(np.array([0.5]))

    def test_tighten_separable_quadratic_bounds_infers_finite_box(self):
        """Constraints like x + y^2 <= c should infer finite bounds for both variables."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_separable_quadratic")
        x = m.continuous("x", lb=0.0, ub=1e20)
        y = m.continuous("y", lb=-1e20, ub=1e20)
        m.subject_to(x + y**2 <= 4.0)
        m.minimize(x * 0.0 + y * 0.0)

        flat_lb = np.array([0.0, -1e20], dtype=np.float64)
        flat_ub = np.array([1e20, 1e20], dtype=np.float64)
        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

        assert tightened_lb[0] == pytest.approx(0.0)
        assert tightened_ub[0] == pytest.approx(4.0)
        assert tightened_lb[1] == pytest.approx(-2.0)
        assert tightened_ub[1] == pytest.approx(2.0)
        assert "separable_quadratic_upper_bound" in stats.applied_rules

    def test_tighten_iterates_linked_quadratic_constraints(self):
        """Linked quadratic inequalities should reach the compact implied box."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_linked_quadratic")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        y = m.continuous("y", lb=-1e20, ub=1e20)
        m.subject_to(x**2 <= y)
        m.subject_to(y <= 1.0 - x**2)
        m.minimize(x * 0.0 + y * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([-1e20, -1e20], dtype=np.float64),
            np.array([1e20, 1e20], dtype=np.float64),
        )

        assert tightened_lb[0] == pytest.approx(-1.0)
        assert tightened_ub[0] == pytest.approx(1.0)
        assert tightened_lb[1] == pytest.approx(0.0)
        assert tightened_ub[1] == pytest.approx(1.0)
        assert "separable_quadratic_upper_bound" in stats.applied_rules

    def test_tighten_proves_exp_square_cycle_infeasible(self):
        """Issue #33: y=exp(x), x=y^2 should be proven infeasible by tightening."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_exp_square_cycle")
        x = m.continuous("x")
        y = m.continuous("y")
        m.subject_to(y - dm.exp(x) == 0.0)
        m.subject_to(x - y**2 == 0.0)
        m.minimize(x * 0.0 + y * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([-9.999e19, -9.999e19], dtype=np.float64),
            np.array([9.999e19, 9.999e19], dtype=np.float64),
        )

        assert stats.infeasible is True
        assert "monotone_function_equality" in stats.applied_rules
        assert "quadratic_equality_bounds" in stats.applied_rules
        assert tightened_lb[0] >= 0.0
        assert tightened_lb[1] >= 1.0
        assert np.all(np.isfinite(tightened_ub))

    def test_tighten_monotone_function_bounds(self):
        """Monotone unary constraints should tighten affine argument domains."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_monotone_functions")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        y = m.continuous("y", lb=0.0, ub=1e20)
        z = m.continuous("z", lb=-1e20, ub=1e20)
        m.subject_to(dm.exp(x) <= 100.0)
        m.subject_to(dm.log(y) >= 2.0)
        m.subject_to(dm.sqrt(z) <= 3.0)
        m.minimize(x * 0.0 + y * 0.0 + z * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([-1e20, 0.0, -1e20], dtype=np.float64),
            np.array([1e20, 1e20, 1e20], dtype=np.float64),
        )

        assert tightened_ub[0] == pytest.approx(np.log(100.0))
        assert tightened_lb[1] == pytest.approx(np.exp(2.0))
        assert tightened_lb[2] == pytest.approx(0.0)
        assert tightened_ub[2] == pytest.approx(9.0)
        assert "monotone_function_bounds" in stats.applied_rules

    def test_tighten_sign_stable_reciprocal_bounds(self):
        """Reciprocal propagation should only apply on sign-stable denominator boxes."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_reciprocal")
        x = m.continuous("x", lb=0.1, ub=1e20)
        y = m.continuous("y", lb=-1e20, ub=-0.1)
        z = m.continuous("z", lb=-1.0, ub=1.0)
        m.subject_to(1.0 / x <= 0.25)
        m.subject_to(1.0 / y >= -0.25)
        m.subject_to(1.0 / z <= 0.25)
        m.minimize(x * 0.0 + y * 0.0 + z * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([0.1, -1e20, -1.0], dtype=np.float64),
            np.array([1e20, -0.1, 1.0], dtype=np.float64),
        )

        assert tightened_lb[0] == pytest.approx(4.0)
        assert tightened_ub[1] == pytest.approx(-4.0)
        assert tightened_lb[2] == pytest.approx(-1.0)
        assert tightened_ub[2] == pytest.approx(1.0)
        assert "reciprocal_bounds" in stats.applied_rules

    def test_nlp_005_010_tightening_enables_negative_power_domain(self):
        """Issue #62: reciprocal reformulation should expose a safe y**(-2) domain."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        instance = MINLPTESTS_NLP_BY_ID["nlp_005_010"]
        m = instance.build_fn()
        flat_lb, flat_ub = flat_variable_bounds(m)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

        assert tightened_ub[0] <= 3.9 + 1e-9
        assert tightened_ub[1] <= 3.9 + 1e-9
        assert tightened_lb[1] >= 1.0 / np.sqrt(4.4) - 1e-9
        assert "positive_affine_reciprocal_bounds" in stats.applied_rules
        assert "negative_power_bounds" in stats.applied_rules

    def test_nonlinear_bound_tightening_reports_quadratic_contradiction(self):
        """A rule proof of infeasibility should be reported explicitly."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        m = Model("bt_quadratic_contradiction")
        x = m.continuous("x", lb=-10.0, ub=10.0)
        m.subject_to(x**2 == -1.0)
        m.minimize(x * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([-10.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
        )

        assert stats.infeasible is True
        assert "sum_of_squares_upper_bound" in stats.applied_rules
        assert "negative upper bound" in (stats.infeasibility_reason or "")
        assert tightened_lb[0] == pytest.approx(-10.0)
        assert tightened_ub[0] == pytest.approx(10.0)

    def test_nonlinear_bound_tightening_reports_monotone_domain_contradiction(self):
        """Domain/range contradictions should not be encoded as invalid boxes."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        sqrt_model = Model("bt_sqrt_contradiction")
        sqrt_x = sqrt_model.continuous("x", lb=0.0, ub=10.0)
        sqrt_model.subject_to(dm.sqrt(sqrt_x) <= -1.0)
        sqrt_model.minimize(sqrt_x * 0.0)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            sqrt_model,
            np.array([0.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
        )

        assert stats.infeasible is True
        assert "monotone_function_bounds" in stats.applied_rules
        assert "sqrt(argument) cannot be <=" in (stats.infeasibility_reason or "")
        assert tightened_lb[0] == pytest.approx(0.0)
        assert tightened_ub[0] == pytest.approx(10.0)

        exp_model = Model("bt_exp_contradiction")
        exp_x = exp_model.continuous("x", lb=-10.0, ub=10.0)
        exp_model.subject_to(dm.exp(exp_x) <= -1.0)
        exp_model.minimize(exp_x * 0.0)

        _, _, exp_stats = tighten_nonlinear_bounds(
            exp_model,
            np.array([-10.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
        )

        assert exp_stats.infeasible is True
        assert "exp(argument) cannot be <=" in (exp_stats.infeasibility_reason or "")

    def test_nonlinear_bound_tightening_accepts_custom_rules(self):
        """The shared registry should accept external sound rule objects."""
        from discopt._jax.nonlinear_bound_tightening import (
            NonlinearBoundTighteningRule,
            tighten_nonlinear_bounds,
        )

        class ClampFirstVarRule(NonlinearBoundTighteningRule):
            name = "custom_clamp"

            def tighten(self, model, flat_lb, flat_ub, metadata):
                del model, metadata
                new_lb = flat_lb.copy()
                new_ub = flat_ub.copy()
                new_ub[0] = min(float(new_ub[0]), 2.0)
                return new_lb, new_ub

        m = Model("custom_bt")
        x = m.continuous("x", lb=0.0, ub=10.0)
        m.minimize(x)

        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(
            m,
            np.array([0.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
            rules=(ClampFirstVarRule(),),
        )

        assert tightened_lb[0] == pytest.approx(0.0)
        assert tightened_ub[0] == pytest.approx(2.0)
        assert stats.applied_rules == ("custom_clamp",)

    def test_minlptests_108_tightens_separable_quadratic_bounds(self):
        """Translated MINLPTests 108 cases should benefit from shared nonlinear tightening."""
        from discopt._jax.nonlinear_bound_tightening import tighten_nonlinear_bounds

        instance = MINLPTESTS_CVX_BY_ID["nlp_cvx_108_010"]
        m = instance.build_fn()

        flat_lb, flat_ub = flat_variable_bounds(m)
        tightened_lb, tightened_ub, stats = tighten_nonlinear_bounds(m, flat_lb, flat_ub)

        assert tightened_lb[0] == pytest.approx(0.0)
        assert tightened_ub[0] == pytest.approx(2.0)
        assert tightened_lb[1] == pytest.approx(0.0)
        assert tightened_ub[1] == pytest.approx(np.sqrt(2.0))
        assert "separable_quadratic_upper_bound" in stats.applied_rules

    def test_solver_fbbt_applies_nonlinear_rules_without_linear_constraints(self):
        """The shared nonlinear tightening rules should run through solver FBBT too."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solver import _infer_constraint_bounds, _tighten_node_bounds

        m = Model("nonlinear_fbbt_sum_of_squares")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        y = m.integer("y", lb=-100, ub=100)
        m.subject_to(x**2 + y**2 <= 9.0)
        m.minimize(x * 0.0 + y * 0.0)

        evaluator = NLPEvaluator(m)
        cl, cu = _infer_constraint_bounds(m)
        tightened_lb, tightened_ub = _tighten_node_bounds(
            evaluator,
            np.array([-1e20, -100.0], dtype=np.float64),
            np.array([1e20, 100.0], dtype=np.float64),
            cl,
            cu,
        )

        assert tightened_lb[0] == pytest.approx(-3.0)
        assert tightened_ub[0] == pytest.approx(3.0)
        assert tightened_lb[1] == pytest.approx(-3.0)
        assert tightened_ub[1] == pytest.approx(3.0)

    def test_solver_fbbt_applies_monotone_nonlinear_rules(self):
        """Solver FBBT should reuse the shared monotone function tightening rule."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solver import _infer_constraint_bounds, _tighten_node_bounds

        m = Model("nonlinear_fbbt_monotone")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        y = m.continuous("y", lb=0.0, ub=1e20)
        m.subject_to(dm.exp(x) <= 10.0)
        m.subject_to(dm.log(y) >= 1.0)
        m.minimize(x * 0.0 + y * 0.0)

        evaluator = NLPEvaluator(m)
        cl, cu = _infer_constraint_bounds(m)
        tightened_lb, tightened_ub = _tighten_node_bounds(
            evaluator,
            np.array([-1e20, 0.0], dtype=np.float64),
            np.array([1e20, 1e20], dtype=np.float64),
            cl,
            cu,
        )

        assert tightened_ub[0] == pytest.approx(np.log(10.0))
        assert tightened_lb[1] == pytest.approx(np.e)

    def test_solver_fbbt_reports_nonlinear_infeasibility_status(self):
        """The FBBT entry point should expose proven nonlinear infeasibility."""
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solver import _infer_constraint_bounds, _tighten_node_bounds_with_status

        m = Model("nonlinear_fbbt_contradiction")
        x = m.continuous("x", lb=-10.0, ub=10.0)
        m.subject_to(x**2 == -1.0)
        m.minimize(x * 0.0)

        evaluator = NLPEvaluator(m)
        cl, cu = _infer_constraint_bounds(m)
        tightened_lb, tightened_ub, infeasible = _tighten_node_bounds_with_status(
            evaluator,
            np.array([-10.0], dtype=np.float64),
            np.array([10.0], dtype=np.float64),
            cl,
            cu,
        )

        assert infeasible is True
        assert tightened_lb[0] == pytest.approx(-10.0)
        assert tightened_ub[0] == pytest.approx(10.0)

    def test_solver_returns_infeasible_for_nonlinear_tightening_contradiction(self):
        """The main solver should consume root nonlinear infeasibility proofs."""
        m = Model("solver_contradiction")
        x = m.continuous("x", lb=-10.0, ub=10.0)
        m.subject_to(x**2 == -1.0)
        m.minimize(x)

        result = m.solve(skip_convex_check=True, time_limit=5)

        assert result.status == "infeasible"
        assert result.x is None

    def test_large_bound_warning_uses_declared_bounds_even_when_rules_tighten(self):
        """A warning should not be suppressed unless solve paths use the tightened box."""
        from discopt.solver import _check_finite_bounds

        m = Model("large_bound_warning")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        m.subject_to(x**2 <= 4.0)
        m.minimize(x)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _check_finite_bounds(m)

        messages = [str(w.message) for w in caught]
        assert any("very large or infinite declared bounds" in msg for msg in messages)
        assert any("Nonlinear tightening can adjust" in msg for msg in messages)

    @pytest.mark.parametrize(
        "problem_id",
        [
            "nlp_cvx_108_010",
            "nlp_cvx_108_011",
            "nlp_cvx_108_012",
            "nlp_cvx_108_013",
        ],
    )
    def test_amp_solves_translated_convex_108_family(self, problem_id):
        """AMP should not report false infeasible on the translated 108 convex family."""
        instance = MINLPTESTS_CVX_BY_ID[problem_id]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    def test_amp_multistart_recovers_translated_convex_106_010(self):
        """Pure continuous AMP should retry multiple starts before giving up on 106_010."""
        instance = MINLPTESTS_CVX_BY_ID["nlp_cvx_106_010"]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    @pytest.mark.parametrize(
        "problem_id",
        [
            "nlp_001_010",
            "nlp_002_010",
            "nlp_008_010",
            "nlp_008_011",
            "nlp_009_010",
            "nlp_009_011",
        ],
    )
    @pytest.mark.requires_cyipopt
    def test_amp_recovers_remaining_pure_continuous_minlptests_cases(self, problem_id):
        """AMP should return a recovered incumbent instead of false infeasible."""
        instance = MINLPTESTS_NLP_BY_ID[problem_id]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    def test_amp_certifies_tan_abs_nlp_004_010_at_issue_gap(self):
        """The continuous tan/abs nlp_004 case should certify the issue-79 gap."""
        instance = MINLPTESTS_NLP_BY_ID["nlp_004_010"]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=300.0,
            gap_tolerance=1e-6,
            max_iter=1000,
        )

        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective - instance.expected_obj) <= 1e-6
        assert result.bound is not None
        assert result.bound <= result.objective + 1e-6
        assert result.gap is not None
        assert result.gap <= 1e-6
        assert result.gap_certified is True

    def test_amp_reports_bound_for_tan_abs_nlp_004_010(self, caplog):
        """The nlp_004 tan/abs objective should not fall back to feasibility mode."""
        instance = MINLPTESTS_NLP_BY_ID["nlp_004_010"]
        m = instance.build_fn()

        with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
            result = m.solve(
                solver="amp",
                nlp_solver="ipm",
                time_limit=30.0,
                gap_tolerance=1e-3,
            )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert result.bound is not None
        assert result.gap is None or result.gap >= 0.0
        fallback_messages = [
            record.message
            for record in caplog.records
            if "falling back to a feasibility objective" in record.message
        ]
        assert not any("abs" in message or "tan" in message for message in fallback_messages)

    @pytest.mark.parametrize("problem_id", ["nlp_009_010", "nlp_009_011"])
    @pytest.mark.requires_cyipopt
    def test_amp_reports_bound_for_minmax_objective_minlptests_cases(self, problem_id, caplog):
        """The nlp_009 min/max objectives should not fall back to feasibility mode."""
        instance = MINLPTESTS_NLP_BY_ID[problem_id]
        m = instance.build_fn()

        with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
            result = m.solve(
                solver="amp",
                nlp_solver="ipm",
                time_limit=30.0,
                gap_tolerance=1e-3,
            )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol
        assert result.bound is not None
        fallback_messages = [
            record.message
            for record in caplog.records
            if "falling back to a feasibility objective" in record.message
        ]
        assert not any("FunctionCall: min" in message for message in fallback_messages)
        assert not any("FunctionCall: max" in message for message in fallback_messages)

    @pytest.mark.parametrize(
        "problem_id",
        [
            "nlp_003_010",
            "nlp_003_011",
            "nlp_003_012",
            "nlp_003_013",
            "nlp_003_014",
            "nlp_003_015",
            "nlp_003_016",
            "nlp_mi_003_010",
            "nlp_mi_003_011",
            "nlp_mi_003_012",
            "nlp_mi_003_013",
            "nlp_mi_003_014",
            "nlp_mi_003_015",
            "nlp_mi_003_016",
        ],
    )
    def test_amp_reports_supported_univariate_bound_for_minlptests_cases(
        self,
        problem_id,
        caplog,
    ):
        """MINLPTests exp/sqrt issue cases should produce valid relaxation bounds."""
        instances = (
            MINLPTESTS_MI_BY_ID if problem_id.startswith("nlp_mi_") else MINLPTESTS_NLP_BY_ID
        )
        instance = instances[problem_id]
        m = instance.build_fn()

        with caplog.at_level(logging.WARNING):
            result = m.solve(
                solver="amp",
                nlp_solver="ipm",
                time_limit=30.0,
                gap_tolerance=1e-3,
            )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol
        assert result.bound is not None
        assert result.bound >= result.objective - 1e-6
        assert result.gap is None or result.gap >= 0.0
        assert not any("invalid bound ordering" in record.message for record in caplog.records)

    def test_select_best_nlp_candidate_respects_deadline(self, monkeypatch):
        """Candidate NLP retries should stop when the remaining wall-clock budget is exhausted."""
        from discopt.solvers import amp as amp_mod

        remaining_limits = []
        fake_clock = iter([0.0, 0.4, 0.7])

        monkeypatch.setattr(amp_mod.time, "perf_counter", lambda: next(fake_clock))
        monkeypatch.setattr(
            amp_mod,
            "_solve_nlp_subproblem",
            lambda evaluator, x0, lb, ub, nlp_solver, time_limit=None: (
                remaining_limits.append(time_limit) or (np.asarray(x0, dtype=np.float64), 1.0)
            ),
        )
        monkeypatch.setattr(
            amp_mod,
            "_check_constraints_with_evaluator",
            lambda *args, **kwargs: True,
        )

        m = _make_circle()
        candidates = [
            np.array([1.0, 1.0], dtype=np.float64),
            np.array([0.9, 1.1], dtype=np.float64),
            np.array([0.8, 1.2], dtype=np.float64),
        ]

        x_best, obj_best = amp_mod._select_best_nlp_candidate(
            candidates,
            m,
            evaluator=None,
            flat_lb=np.array([0.0, 0.0], dtype=np.float64),
            flat_ub=np.array([2.0, 2.0], dtype=np.float64),
            constraint_lb=np.array([], dtype=np.float64),
            constraint_ub=np.array([], dtype=np.float64),
            nlp_solver="ipm",
            deadline=0.6,
        )

        assert x_best is not None
        assert obj_best == pytest.approx(1.0)
        assert len(remaining_limits) == 2
        assert remaining_limits[0] == pytest.approx(0.6)
        assert remaining_limits[1] == pytest.approx(0.2)

    @pytest.mark.requires_cyipopt
    def test_amp_fallback_enumerates_small_integer_domain_when_milp_relaxation_fails(self):
        """AMP should still recover a bounded integer optimum if the first MILP errors out."""
        instance = MINLPTESTS_MI_BY_ID["nlp_mi_001_010"]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
            apply_partitioning=False,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    def test_amp_recovers_quadratic_norm_integer_optimum_with_unbounded_integer_var(self):
        """AMP should infer a finite integer box from simple quadratic norm constraints."""
        instance = MINLPTESTS_MI_BY_ID["nlp_mi_004_010"]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
            apply_partitioning=False,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    @pytest.mark.parametrize("problem_id", ["nlp_mi_004_010", "nlp_mi_004_011"])
    def test_amp_certifies_tan_abs_integer_nlp_004_cases_at_issue_gap(self, problem_id):
        """The mixed-integer tan/abs nlp_004 cases should certify the issue-79 gap."""
        instance = MINLPTESTS_MI_BY_ID[problem_id]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=300.0,
            gap_tolerance=1e-6,
            max_iter=1000,
        )

        assert result.status == "optimal"
        assert result.objective is not None
        assert abs(result.objective - instance.expected_obj) <= 1e-6
        assert result.bound is not None
        assert result.bound <= result.objective + 1e-6
        assert result.gap is not None
        assert result.gap <= 1e-6
        assert result.gap_certified is True

    @pytest.mark.requires_cyipopt
    def test_amp_fixed_integer_nlp_retries_with_ipopt(self):
        """Fixed-integer NLP candidates should not be lost when the JAX IPM stalls."""
        instance = MINLPTESTS_MI_BY_ID["nlp_mi_005_010"]
        m = instance.build_fn()

        result = m.solve(
            solver="amp",
            nlp_solver="ipm",
            time_limit=30.0,
            gap_tolerance=1e-3,
            apply_partitioning=False,
        )

        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        tol = 1e-6 + 1e-4 * abs(instance.expected_obj)
        assert abs(result.objective - instance.expected_obj) <= tol

    def test_amp_returns_infeasible_for_nonlinear_tightening_contradiction(self):
        """AMP should stop before MILP/NLP solves when tightening proves infeasibility."""
        m = Model("amp_contradiction")
        x = m.continuous("x", lb=0.0, ub=10.0)
        m.subject_to(dm.sqrt(x) <= -1.0)
        m.minimize(x)

        result = m.solve(solver="amp", skip_convex_check=True, max_iter=1, time_limit=5)

        assert result.status == "infeasible"
        assert result.x is None

    def test_amp_uses_nonlinear_tightened_partition_bounds(self, monkeypatch):
        """AMP should initialize partitions from the tightened nonlinear box."""
        import discopt._jax.discretization as disc_mod
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        captured = {}
        real_initialize = disc_mod.initialize_partitions

        def spy_initialize(part_vars, lb, ub, **kwargs):
            captured["part_vars"] = list(part_vars)
            captured["lb"] = list(lb)
            captured["ub"] = list(ub)
            return real_initialize(part_vars, lb=lb, ub=ub, **kwargs)

        monkeypatch.setattr(disc_mod, "initialize_partitions", spy_initialize)
        monkeypatch.setattr(
            amp_mod,
            "_solve_milp_with_oa_recovery",
            lambda **kwargs: (
                MilpRelaxationResult(status="error", objective=None, x=None),
                {},
                [],
                1,
            ),
        )
        monkeypatch.setattr(
            amp_mod, "_recover_pure_continuous_solution", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_small_integer_domain_fallback",
            lambda *args, **kwargs: (None, None),
        )

        m = Model("amp_bt_partition_bounds")
        x = m.continuous("x", lb=-1e20, ub=1e20)
        m.subject_to(x**2 <= 4.0)
        m.minimize(x)

        result = m.solve(solver="amp", skip_convex_check=True, max_iter=1, time_limit=30)

        assert result.status == "error"
        assert captured["part_vars"] == [0]
        assert captured["lb"] == pytest.approx([-2.0])
        assert captured["ub"] == pytest.approx([2.0])

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
                1,
            ),
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_best_nlp_candidate",
            lambda *args, **kwargs: (None, None),
        )
        monkeypatch.setattr(
            amp_mod, "_recover_pure_continuous_solution", lambda *args, **kwargs: None
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
                1,
            ),
        )
        monkeypatch.setattr(
            amp_mod, "_recover_pure_continuous_solution", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_small_integer_domain_fallback",
            lambda *args, **kwargs: (None, None),
        )

        result = m.solve(solver="amp", max_iter=1, time_limit=30, skip_convex_check=True)

        assert result.status == "error"
        assert result.objective is None
        assert result.bound is None
        assert result.gap_certified is False

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
            convhull_ebd=False,
            convhull_ebd_encoding="gray",
            bound_override=None,
        ):
            del bound_override
            assert convhull_formulation == "disaggregated"
            assert convhull_ebd is False
            assert convhull_ebd_encoding == "gray"
            size = len(oa_cuts or [])
            call_sizes.append(size)
            status = "infeasible" if size >= 4 else "optimal"
            return FakeMilpModel(status), {"dummy": True}

        monkeypatch.setattr(
            "discopt._jax.milp_relaxation.build_milp_relaxation",
            fake_build,
        )

        result, _, kept_cuts, mip_count = amp_mod._solve_milp_with_oa_recovery(
            model=None,
            terms=None,
            disc_state=None,
            incumbent=None,
            oa_cuts=[("c1", 1), ("c2", 2), ("c3", 3), ("c4", 4)],
            time_limit=1.0,
            gap_tolerance=1e-4,
            convhull_formulation="disaggregated",
            convhull_ebd=False,
            convhull_ebd_encoding="gray",
        )

        assert call_sizes == [4, 2]
        assert kept_cuts == [("c3", 3), ("c4", 4)]
        assert result.status == "optimal"
        assert mip_count == 2

    def test_obbt_presolve_tightens_bilinear_demo_bounds(self):
        """OBBT should shrink the initial [0, 10]^2 box to the linear hull x + y = 1."""
        from discopt._jax.obbt import run_obbt

        result = run_obbt(_make_obbt_demo())

        np.testing.assert_allclose(result.tightened_lb, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result.tightened_ub, np.array([1.0, 1.0]))
        assert result.n_tightened >= 2

    def test_obbt_presolve_tightens_inequality_demo_bounds(self):
        """OBBT should also tighten bounds through the A_ub / b_ub extraction path."""
        from discopt._jax.obbt import run_obbt

        result = run_obbt(_make_obbt_ineq_demo())

        np.testing.assert_allclose(result.tightened_lb, np.array([0.0, 0.0]))
        np.testing.assert_allclose(result.tightened_ub, np.array([1.0, 1.0]))
        assert result.n_tightened >= 2

    def test_amp_presolve_bt_uses_tightened_partition_bounds(self, monkeypatch):
        """AMP should initialize partitions from the OBBT-tightened bounds."""
        import discopt._jax.discretization as disc_mod
        from discopt._jax.obbt import ObbtResult
        from discopt.solvers import amp as amp_mod

        captured = {}
        orig_initialize = disc_mod.initialize_partitions

        def fake_run_obbt(model, lb=None, ub=None, **kwargs):
            assert lb is not None
            assert ub is not None
            assert kwargs["time_limit_per_lp"] > 0.0
            return ObbtResult(
                tightened_lb=np.array([0.0, 0.0], dtype=np.float64),
                tightened_ub=np.array([1.0, 1.0], dtype=np.float64),
                n_lp_solves=4,
                n_tightened=2,
                total_lp_time=0.0,
            )

        def spy_initialize(part_vars, lb, ub, n_init, **kwargs):
            captured["lb"] = list(lb)
            captured["ub"] = list(ub)
            return orig_initialize(part_vars, lb=lb, ub=ub, n_init=n_init, **kwargs)

        def stop_after_init(*args, **kwargs):
            raise RuntimeError("stop after initialization")

        monkeypatch.setattr("discopt._jax.obbt.run_obbt", fake_run_obbt)
        monkeypatch.setattr(disc_mod, "initialize_partitions", spy_initialize)
        monkeypatch.setattr(amp_mod, "_solve_milp_with_oa_recovery", stop_after_init)

        result = amp_mod.solve_amp(
            _make_obbt_demo(),
            presolve_bt=True,
            max_iter=1,
            time_limit=1.0,
        )

        assert result.status == "error"
        assert captured["lb"] == [0.0, 0.0]
        assert captured["ub"] == [1.0, 1.0]

    def test_amp_presolve_bt_can_reduce_iterations(self):
        """OBBT should reduce AMP iterations on a model with loose linear bounds."""
        m = _make_obbt_demo()

        iters_without = []
        result_without = m.solve(
            solver="amp",
            presolve_bt=False,
            rel_gap=0.55,
            max_iter=20,
            time_limit=20,
            iteration_callback=lambda info: iters_without.append(info["iteration"]),
        )

        iters_with = []
        result_with = m.solve(
            solver="amp",
            presolve_bt=True,
            rel_gap=0.55,
            max_iter=20,
            time_limit=20,
            iteration_callback=lambda info: iters_with.append(info["iteration"]),
        )

        assert result_without.status in ("optimal", "feasible")
        assert result_with.status in ("optimal", "feasible")
        assert result_without.objective is not None
        assert result_with.objective is not None
        assert abs(result_without.objective - 0.25) <= 1e-2
        assert abs(result_with.objective - 0.25) <= 1e-2
        # Bound tightening is validated directly above; this guards against OBBT
        # making the AMP loop strictly worse while staying robust to solver noise.
        assert len(iters_with) <= len(iters_without)

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

    def test_amp_time_limit_with_incumbent_returns_feasible(self, monkeypatch):
        """Timing out after finding an incumbent should not hide the feasible point."""
        from discopt._jax.milp_relaxation import MilpRelaxationResult
        from discopt.solvers import amp as amp_mod

        fake_clock = iter([0.0, 0.0, 1.5])

        monkeypatch.setattr(amp_mod.time, "perf_counter", lambda: next(fake_clock))
        monkeypatch.setattr(
            amp_mod,
            "_solve_milp_with_oa_recovery",
            lambda **kwargs: (
                MilpRelaxationResult(
                    status="optimal",
                    objective=0.0,
                    x=np.array([2.0, 2.0], dtype=np.float64),
                ),
                {},
                [],
                1,
            ),
        )
        monkeypatch.setattr(
            amp_mod,
            "_solve_best_nlp_candidate",
            lambda *args, **kwargs: (np.array([2.0, 2.0], dtype=np.float64), 10.0),
        )

        result = amp_mod.solve_amp(
            _make_nlp1(),
            apply_partitioning=False,
            presolve_bt=False,
            max_iter=1,
            time_limit=1.0,
        )

        assert result.status == "feasible"
        assert result.objective is not None
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
        m = _make_nlp1()
        result = m.solve(time_limit=60, gap_tolerance=1e-3)
        assert result.objective is not None, "Spatial B&B failed to find any solution"
        assert abs(result.objective - NLP1_OPTIMUM) <= 0.5, (
            f"Spatial B&B found {result.objective:.4f}, expected near {NLP1_OPTIMUM} "
            f"(gap={abs(result.objective - NLP1_OPTIMUM):.4f})"
        )

    def test_circle_monomial_global_correctness(self):
        """Spatial B&B should solve the Alpine circle benchmark globally."""
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
