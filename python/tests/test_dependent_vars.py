"""Tests for the functionally-dependent-variable detector (``_relax/dependent_vars``).

Two things are under test:

*Semantics* — the isolated-affine-coefficient analyzer must keep abstaining
conservatively. Every structural case that made it return ``None`` before issue
#1104's rewrite (nonlinear occurrence, indexed reference, reduction, opaque
node) must still return ``None``, and the affine cases must still return the
same coefficient. The detector only changes *branch order* (see the module
docstring), so a wrong answer is not unsound — but it is wrong, and the rewrite
replaced a per-name recursive walk with a single top-down pass, which is exactly
the kind of change that silently drops a case.

*Cost* — the pass runs before branch-and-bound arms ``time_limit``, so its cost
is part of the user's budget. The pre-#1104 implementation re-derived variable
occurrence at every recursion level for every candidate name, i.e.
``O(names x nodes^2)``: on ``t1000`` (1002 candidates, a 5003-node depth-1004
sum chain) the scan did not return in 12 minutes against a 30 s limit. The
scaling test below builds that structure directly.
"""

from __future__ import annotations

import time

import discopt._relax.dependent_vars as dv
import discopt.modeling as dm
import pytest
from discopt._relax.dependent_vars import (
    _body_is_nonlinear,
    _isolated_affine_coeff,
    _isolated_affine_coeffs,
    dependent_columns_for_model,
    find_functionally_dependent_names,
)
from discopt._relax.objective_epigraph import (
    VarNameIndex,
    WorkCounter,
    _collect_var_names,
    _occurs,
)


@pytest.fixture
def xyz():
    m = dm.Model("t")
    return m, m.continuous("x", lb=-5, ub=5), m.continuous("y", lb=1, ub=5)


@pytest.mark.unit
class TestVarNameIndex:
    def test_names_and_occurrence(self, xyz):
        _, x, y = xyz
        e = 3.0 * x + y * y
        idx = VarNameIndex(e)
        assert idx.names(e) == {"x", "y"}
        assert idx.occurs(e, "x") is True
        assert idx.occurs(e, "z") is False

    def test_shared_subexpression_memoized_once(self, xyz):
        """A DAG node reachable twice is indexed once, not re-expanded."""
        _, x, y = xyz
        shared = x * y + x
        e = shared + shared
        idx = VarNameIndex(e)
        # left and right of the top-level ``+`` are the *same* object.
        assert e.left is e.right is shared
        assert idx.names(shared) == {"x", "y"}
        # One entry per distinct node: the shared subtree contributes its nodes once.
        assert len(idx._memo) == len(VarNameIndex(shared)._memo) + 1

    def test_foreign_node_raises_rather_than_reporting_absent(self, xyz):
        """A node from another DAG must not silently read as variable-free."""
        _, x, y = xyz
        idx = VarNameIndex(x + 1.0)
        with pytest.raises(KeyError):
            idx.names(y * y)

    def test_deep_chain_does_not_exhaust_the_recursion_limit(self, xyz):
        m, x, _ = xyz
        terms = [m.continuous(f"v{i}", lb=0, ub=1) for i in range(3000)]
        e = terms[0]
        for t in terms[1:]:
            e = e + t
        assert len(_collect_var_names(e)) == 3000
        assert _occurs(e, "v2999") is True


@pytest.mark.unit
class TestIsolatedAffineCoefficients:
    """Structural cases: affine ones carry a coefficient, the rest abstain."""

    @pytest.mark.parametrize(
        "build, expected",
        [
            (lambda x, y: x, 1.0),
            (lambda x, y: -x, -1.0),
            (lambda x, y: 3.0 * x, 3.0),
            (lambda x, y: x * 3.0, 3.0),
            (lambda x, y: x / 4.0, 0.25),
            (lambda x, y: x + x, 2.0),
            (lambda x, y: x - x, 0.0),
            (lambda x, y: 2.0 * x - 5.0 * x, -3.0),
            # A sibling term arbitrarily nonlinear in *other* variables does not
            # spoil the target's affine isolation -- the whole point of this
            # analyzer over a whole-expression affine test.
            (lambda x, y: x - dm.sqrt(y * y + 1.0), 1.0),
            (lambda x, y: x + 4243.28 / (y * y), 1.0),
            (lambda x, y: -(2.0 * x) + dm.exp(y), -2.0),
            # Provably absent -> 0.0, never None.
            (lambda x, y: y * y + dm.log(y), 0.0),
            # Nonlinear in the target -> abstain.
            (lambda x, y: x * x, None),
            (lambda x, y: x * y, None),
            (lambda x, y: x**2, None),
            (lambda x, y: 1.0 / x, None),
            (lambda x, y: y / x, None),
            (lambda x, y: dm.exp(x), None),
            (lambda x, y: dm.sqrt(x + 1.0), None),
            (lambda x, y: x + dm.sin(x), None),
            # A non-constant multiplier is not a coefficient.
            (lambda x, y: (y + 1.0) * x, None),
        ],
    )
    def test_cases(self, xyz, build, expected):
        _, x, y = xyz
        e = build(x, y)
        assert _isolated_affine_coeff(e, "x") == expected

    def test_bulk_pass_agrees_with_the_single_name_view(self, xyz):
        _, x, y = xyz
        e = 3.0 * x - dm.sqrt(y) + x
        coeffs = _isolated_affine_coeffs(e)
        assert coeffs == {"x": 4.0, "y": None}
        for name in ("x", "y", "absent"):
            assert _isolated_affine_coeff(e, name) == coeffs.get(name, 0.0)

    def test_shared_subexpression_blowup_abstains_deterministically(self, xyz):
        """A DAG with exponentially many paths must abstain, not hang.

        The multiplier a node inherits is path-dependent, so the traversal is
        O(paths); ``e = e + e`` twenty-five times is 28 distinct nodes and 2**25
        paths. The visit cap turns that into a conservative ``None`` (nothing
        proven) in milliseconds. The cap counts node visits, not seconds, so the
        answer is identical on a fast and a slow machine.
        """
        _, x, _ = xyz
        e = x + 0.0
        for _ in range(25):
            e = e + e
        assert len(VarNameIndex(e)) < 100, "the DAG must stay small; only paths blow up"
        t0 = time.perf_counter()
        assert _isolated_affine_coeffs(e) is None
        assert time.perf_counter() - t0 < 5.0

    def test_deep_chain_is_analyzed_iteratively(self, xyz):
        m, _, _ = xyz
        terms = [m.continuous(f"w{i}", lb=0, ub=1) for i in range(3000)]
        e = terms[0]
        for i, t in enumerate(terms[1:], start=1):
            e = e + float(i) * t
        coeffs = _isolated_affine_coeffs(e)
        assert coeffs["w0"] == 1.0
        assert coeffs["w2999"] == 2999.0


@pytest.mark.unit
class TestBodyIsNonlinear:
    @pytest.mark.parametrize(
        "build, expected",
        [
            (lambda x, y: x + 2.0 * y - 3.0, False),
            (lambda x, y: -(x / 2.0) + y, False),
            (lambda x, y: x * y, True),
            (lambda x, y: x / y, True),
            (lambda x, y: x**2, True),
            (lambda x, y: dm.exp(x) + y, True),
            (lambda x, y: x - dm.sqrt(y * y + 1.0), True),
        ],
    )
    def test_cases(self, xyz, build, expected):
        _, x, y = xyz
        assert _body_is_nonlinear(build(x, y)) is expected


@pytest.mark.unit
class TestFindFunctionallyDependentNames:
    def test_marks_the_pinned_output_only(self):
        m = dm.Model("pinned")
        x = m.continuous("x", lb=1, ub=5)
        y = m.continuous("y", lb=1, ub=5)
        z = m.continuous("z", lb=-100, ub=100)
        m.subject_to(z - x * y == 0.0)
        m.minimize(z)
        assert find_functionally_dependent_names(m) == {"z"}
        assert dependent_columns_for_model(m, {"z"}) == [2]

    def test_affine_defining_equality_is_left_to_presolve(self):
        m = dm.Model("affine")
        x = m.continuous("x", lb=1, ub=5)
        z = m.continuous("z", lb=-100, ub=100)
        m.subject_to(z - 2.0 * x == 0.0)
        m.minimize(z)
        assert find_functionally_dependent_names(m) == set()

    def test_inequality_is_not_a_defining_equality(self):
        m = dm.Model("ineq")
        x = m.continuous("x", lb=1, ub=5)
        z = m.continuous("z", lb=-100, ub=100)
        m.subject_to(z - x * x <= 0.0)
        m.minimize(z)
        assert find_functionally_dependent_names(m) == set()

    def test_integer_outputs_are_not_candidates(self):
        m = dm.Model("intout")
        x = m.continuous("x", lb=1, ub=5)
        n = m.integer("n", lb=0, ub=10)
        m.subject_to(n - x * x == 0.0)
        m.minimize(x)
        assert find_functionally_dependent_names(m) == set()


def _chain_model(n_terms: int) -> dm.Model:
    """``t1000``-shaped probe: one nonlinear equality over a long sum chain.

    ``sum_i c_i * v_i * v_i + z == 0`` with every ``v_i`` a distinct candidate.
    The pre-#1104 scan tested occurrence for each of the ``n_terms`` candidates
    at every level of the depth-``n_terms`` chain.
    """
    m = dm.Model("chain")
    v = [m.continuous(f"v{i}", lb=0.0, ub=1.0) for i in range(n_terms)]
    z = m.continuous("z", lb=-1e6, ub=1e6)
    body = z
    for i, vi in enumerate(v):
        body = body + float(i + 1) * (vi * vi)
    m.subject_to(body == 0.0)
    m.minimize(z)
    return m


@pytest.mark.unit
class TestScanCost:
    def test_long_chain_scan_is_not_quadratic(self):
        """Issue #1104: the scan must stay tractable on a t1000-shaped body.

        The bound is deliberately loose (measured ~0.03 s on the t1000 model
        itself, ~4000 nodes over 1000 candidates) so machine load cannot flake
        it; the pre-fix implementation did not finish this shape in 12 minutes.
        """
        m = _chain_model(1000)
        t0 = time.perf_counter()
        names = find_functionally_dependent_names(m)
        wall = time.perf_counter() - t0
        assert names == {"z"}, "the pinned output must still be detected"
        assert wall < 10.0, f"scan took {wall:.1f}s on a 1000-term chain"

    def test_exhausted_work_budget_stops_the_scan(self):
        """A blown budget yields the (sound) partial set, not an overrun."""
        m = _chain_model(200)
        assert find_functionally_dependent_names(m) == {"z"}
        assert find_functionally_dependent_names(m, work_budget=0) == set()
        assert find_functionally_dependent_names(m, work_budget=50) == set()

    def test_generous_work_budget_does_not_change_the_answer(self):
        m = _chain_model(200)
        assert find_functionally_dependent_names(m, work_budget=10**9) == {"z"}

    def test_budget_is_deterministic_not_wall_clock(self):
        """The same budget must give the same answer on any machine.

        #912: this set steers spatial branching, so a clock-based cut would make
        the search tree a function of machine speed. The exhaustion point is a
        pure function of the model and the allowance, so the boundary is
        reproducible — this pins that by finding it and re-checking it.
        """
        m = _chain_model(60)
        full = find_functionally_dependent_names(m)
        assert full == {"z"}
        # Bisect for the smallest allowance that still yields the full answer.
        lo, hi = 0, 10**7
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if find_functionally_dependent_names(m, work_budget=mid) == full:
                hi = mid
            else:
                lo = mid
        # Reproducible on both sides of the boundary, every time.
        for _ in range(5):
            assert find_functionally_dependent_names(m, work_budget=hi) == full
            assert find_functionally_dependent_names(m, work_budget=lo) != full

    def test_scan_work_budget_bounds_the_default_path(self):
        """The default allowance is what the solver relies on; it must be finite
        and must be what an un-parameterised call spends."""
        assert 0 < dv._SCAN_WORK_BUDGET < 10**12
        m = _chain_model(50)
        assert find_functionally_dependent_names(m) == find_functionally_dependent_names(
            m, work_budget=dv._SCAN_WORK_BUDGET
        )


@pytest.mark.unit
class TestWorkCounter:
    def test_spend_reports_exhaustion(self):
        w = WorkCounter(10)
        assert w.spend(4) and not w.exhausted
        assert w.spend(5) and not w.exhausted
        assert not w.spend(1) and w.exhausted
        assert not w.spend(1), "an exhausted counter stays exhausted"
        assert w.spent == 10

    def test_exhausted_index_reports_opaque_everywhere(self, xyz):
        """An index that ran out mid-build must not answer from a partial memo.

        Reporting ``None`` (opaque) / ``True`` (might occur) is the sound
        direction; reporting "no variables" from a half-built memo would make
        every occurrence test silently wrong.
        """
        _, x, y = xyz
        expr = x * x + y
        idx = VarNameIndex(expr, work=WorkCounter(1))
        assert idx.exhausted
        assert idx.names(expr) is None
        assert idx.occurs(expr, "x") is True
        assert idx.occurs(expr, "not_even_a_variable") is True
        assert len(idx) == 0

    def test_unexhausted_index_is_unaffected_by_a_generous_counter(self, xyz):
        _, x, y = xyz
        expr = x * x + y
        plain = VarNameIndex(expr)
        metered = VarNameIndex(expr, work=WorkCounter(10**9))
        assert not metered.exhausted
        assert metered.names(expr) == plain.names(expr) == frozenset({"x", "y"})
        assert len(metered) == len(plain)

    def test_index_build_charges_more_than_one_unit_per_node(self, xyz):
        """The charge must track the union work, not just the node count.

        A per-node-only charge would under-count by the factor that actually
        hurts: a 1000-name chain unions ~1e6 elements over ~4000 nodes.
        """
        m = _chain_model(200)
        body = m._constraints[0].body
        w = WorkCounter(10**9)
        idx = VarNameIndex(body, work=w)
        assert not idx.exhausted
        assert w.spent > 10 * len(idx), (
            f"charged {w.spent} units for {len(idx)} nodes — the union work is not being counted"
        )
