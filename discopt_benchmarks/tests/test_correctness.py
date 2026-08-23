"""
Correctness Validation & Verification Tests

These tests verify that discopt produces CORRECT results.
Correctness is the highest-priority property — the solver must
never claim a solution is globally optimal when it is not.

Test categories:
1. Known-optimum validation against MINLPLib
2. Feasibility verification (does the solution satisfy constraints?)
3. Bound validity (is the lower bound actually a lower bound?)
4. Determinism (same result across multiple runs?)
5. Edge cases (infeasible, unbounded, degenerate problems)
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────
# Test fixtures and helpers
# ─────────────────────────────────────────────────────────────
#
# This file shipped as a scaffold: ``solve_instance`` raised
# ``NotImplementedError``, every test caught it and called
# ``pytest.skip("discopt not yet available")``, and all 98 tests skipped —
# including the two in ``TestDeterminism``, the only assertions anywhere in the
# repository that the solver reproduces run to run. #1116 made
# ``deterministic=`` real (it had been a documented-but-unread parameter), and
# those two tests were the flag's only consumer, so the flag would have had no
# consumer that ever executed. That is the CLAUDE.md §6 failure mode — an
# instrument that measures nothing and reports a pass — sitting in the file named
# ``test_correctness.py``.
#
# The skips that remain are now *accurate*: an instance is skipped when its
# ``.nl`` is not on this machine, which is a real condition (the 4 800-instance
# MINLPLib snapshot is a developer-local Dropbox tree), and never a claim about
# discopt itself.

# The in-repo corpus is searched first so CI, which has no MINLPLib snapshot,
# still runs the instances it does ship.
_REPO_NL = pathlib.Path(__file__).resolve().parents[2] / "python/tests/data/minlplib_nl"
_BENCH_NL = pathlib.Path(
    os.environ.get(
        "DISCOPT_MINLPLIB_NL",
        str(pathlib.Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"),
    )
)

# Wall budget per solve. The scaffold asked for 3600 s on 29 instances across four
# test classes, which is not a suite anyone runs. These tests skip when an instance
# does not reach optimality, so a smaller budget narrows *coverage*, never
# correctness — and every assertion here is a correctness assertion, not a timing
# one. Raise it with DISCOPT_BENCH_TIME_LIMIT for a wider local run.
TIME_LIMIT = float(os.environ.get("DISCOPT_BENCH_TIME_LIMIT", "30"))

# These solve real instances, so they belong to the `correctness` marker (which
# the root `addopts` deselects by default) and run on CI's python-correctness
# lane. That lane's per-test timeout is 120 s, which TestDeterminism's three
# sequential solves have to fit inside -- hence 30 s rather than the scaffold's
# 3600 s.
pytestmark = pytest.mark.correctness

# Read by the no-op suite guard in ``conftest.py`` (#1050): this module exists to
# run, so a session in which every one of its tests skipped is a failure, not a
# quiet pass. It is the backstop for ``test_the_corpus_is_reachable`` below —
# that test fails when no instance resolves, this fires even if the reason
# changes to something nobody anticipated.
MUST_EXECUTE = True

_SOLVE_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def _resolve(name: str) -> pathlib.Path:
    for root in (_REPO_NL, _BENCH_NL):
        p = root / f"{name}.nl"
        if p.exists():
            return p
    pytest.skip(f"instance {name}.nl not present (set DISCOPT_MINLPLIB_NL)")


def _load(name: str):
    """Parse an instance. Cached: parsing is pure, so re-reading buys nothing."""
    path = _resolve(name)
    if path not in _MODEL_CACHE:
        from discopt.modeling.core import from_nl

        _MODEL_CACHE[path] = from_nl(str(path))
    return _MODEL_CACHE[path]


def _is_maximize(name: str) -> bool:
    """True when the instance maximizes.

    MINLPLib mixes senses and `.solu` records the optimum without one, so a
    bound check that assumes minimization silently tests the wrong inequality on
    the instances that do not.
    """
    return _load(name)._objective.sense.value == "maximize"


def solve_instance(name: str, *, cache: bool = True, **kwargs):
    """Solve a MINLPLib instance by name and return discopt's ``SolveResult``.

    ``cache=False`` is mandatory for anything measuring run-to-run behaviour:
    a memoised solve would hand the SAME result object to every repetition and
    ``TestDeterminism`` would pass by construction, which is the failure this
    file already had once.
    """
    key = (name, tuple(sorted(kwargs.items())))
    if cache and key in _SOLVE_CACHE:
        return _SOLVE_CACHE[key]
    kwargs.setdefault("time_limit", TIME_LIMIT)
    # A fresh parse per solve: a Model carries solve-time state, so reusing one
    # across repetitions would let a determinism test measure inherited state
    # rather than the solver.
    from discopt.modeling.core import from_nl

    res = from_nl(str(_resolve(name))).solve(**kwargs)
    if cache:
        _SOLVE_CACHE[key] = res
    return res


def _flat_x(name: str, res) -> np.ndarray | None:
    """``SolveResult.x`` is a dict of name -> array; flatten it in model order."""
    if not res.x:
        return None
    parts = []
    for v in _load(name)._variables:
        if v.name not in res.x:
            return None
        parts.append(np.asarray(res.x[v.name], dtype=float).flatten())
    return np.concatenate(parts) if parts else None


# ─────────────────────────────────────────────────────────────
# 1. KNOWN OPTIMUM VALIDATION
# ─────────────────────────────────────────────────────────────

# Known optimal values from MINLPLib (subset for testing).
# Verified against the official MINLPLib `minlplib.solu`; nine entries were
# demonstrably wrong (most ~100-1500x too small) and are corrected here to the
# authoritative =opt= (proven) / =best= (best known) values.
KNOWN_OPTIMA = {
    "ex1221": 7.6672,
    "ex1222": 1.0765,
    "ex1223": 4.5796,
    "ex1223a": 4.5796,
    "ex1224": -0.94347,
    "ex1225": 31.0,  # was 0.0 (=opt=)
    "ex1226": -17.0,
    "ex1233": 155010.6713,  # was 62.1833 (=best=)
    "ex1243": 83402.50641,  # was 83.6455 (=opt=)
    "ex1244": 82042.90522,  # was 83.6455 (=opt=)
    "ex1252": 128893.741,  # was 1169.37 (=opt=; Kocis-Grossmann batch plant)
    "ex1252a": 128893.741,  # was 1169.37 (=opt=)
    "ex1263": 19.6,  # was 19.46 (=opt=)
    "ex1263a": 19.6,  # was 19.46 (=opt=)
    "ex1264": 8.6,
    "ex1264a": 8.6,
    "ex1265": 10.3,
    "ex1265a": 10.3,
    "ex1266": 16.3,
    "ex1266a": 16.3,
    "fuel": 8566.12,
    "gastrans": 89.08588,
    "ghg_1veh": 7.7816348850,  # was -246.04 (=opt=)
    # `procurement1` (0.0) and `smallinvDAXr1b50` (-3.83) were phantom entries —
    # those names match no MINLPLib instance and the values matched nothing.
    # Replaced with the real instances the suites reference (=best= / =opt=).
    "procurement1large": 3802.1797490,
    "procurement1mot": 291.5416577,
    "procurement2mot": 212.0707488,
    # The only maximizing instance with a `.solu` optimum that ships *in this
    # repository* (`python/tests/data/minlplib_nl/`). CI has no access to the
    # MINLPLib snapshot, so without it the sense-aware arm of
    # `test_bound_validity` is exercised on a developer machine and nowhere else
    # -- which is how the sense bug survived #1120 in the first place.
    "syn05hfsg": 837.7324009,
    "smallinvDAXr1b020-022": 1.5715279820,
    "smallinvDAXr1b050-055": 9.7971434540,
    "smallinvDAXr1b100-110": 39.1621418600,
}

ABS_TOL = 1e-4
REL_TOL = 1e-3


def bound_is_valid(bound: float, optimum: float, *, maximize: bool, tol: float = ABS_TOL) -> bool:
    """Does a dual bound respect the true optimum, given the objective sense?

    Minimizing, the dual bound is a *lower* bound and is invalid when it rises
    above the optimum. Maximizing, it is an *upper* bound and is invalid when it
    falls below. This is a pure function so both directions can be tested with
    no corpus, no solve, and no dependence on which `.nl` files a machine has --
    the corpus-based check below cannot do that, which is exactly how the
    minimization-only version shipped.
    """
    return bound >= optimum - tol if maximize else bound <= optimum + tol


# Resolved at collection time so parametrisation can prefer instances that exist.
# `_resolve` cannot be used here -- it calls `pytest.skip`, which is a runtime
# operation.
AVAILABLE = [
    n for n in KNOWN_OPTIMA if (_REPO_NL / f"{n}.nl").exists() or (_BENCH_NL / f"{n}.nl").exists()
]


@pytest.mark.smoke
def test_the_corpus_is_reachable():
    """Guard: without this, a broken corpus path turns 98 tests into 98 skips.

    That is exactly how this file spent its life before #1116 -- green, and
    measuring nothing (CLAUDE.md §6).
    """
    assert AVAILABLE, (
        f"no KNOWN_OPTIMA instance resolved under {_REPO_NL} or {_BENCH_NL}; "
        "every instance-based test below is vacuous"
    )


@pytest.mark.parametrize(
    "bound,optimum,maximize,expected",
    [
        # Minimizing: the dual bound is a LOWER bound.
        (1.0, 2.0, False, True),  # below the optimum -- sound
        (2.0, 2.0, False, True),  # equal -- sound
        (3.0, 2.0, False, False),  # above the optimum -- cuts it off
        # Maximizing: the dual bound is an UPPER bound. These four are the cases
        # the pre-fix check got wrong: it called the first two invalid (they are
        # sound) and the last valid (it cuts the optimum off).
        (3593.688106, 291.5416577, True, True),  # the real procurement1mot bound
        (40237.35099, 3802.179749, True, True),  # the real procurement1large bound
        (291.5416577, 291.5416577, True, True),  # equal -- sound
        (200.0, 291.5416577, True, False),  # BELOW the optimum -- cuts it off
    ],
)
def test_bound_validity_respects_the_objective_sense(bound, optimum, maximize, expected):
    """The direction of the bound check, tested without a corpus or a solve.

    `test_bound_validity` below can only exercise the senses that happen to be
    present on the machine running it, and CI ships only the 66-file in-repo
    corpus. That gap is what let a minimization-only assertion survive #1120, so
    the decision itself is pulled out and tested directly, in both directions,
    everywhere (CLAUDE.md §6).
    """
    assert bound_is_valid(bound, optimum, maximize=maximize) is expected


@pytest.mark.smoke
def test_the_corpus_exposes_a_readable_sense_for_every_instance():
    """Every available instance must yield a sense, and the split is reported.

    Asserting *which* senses exist would encode a property of one machine's
    corpus -- CI resolves far fewer instances than a developer box with the
    MINLPLib snapshot. What must hold everywhere is that the sense is readable
    at all: if `_objective.sense` ever stops being the place the sense lives,
    `_is_maximize` would raise rather than quietly answer "minimize" and send
    every bound check back down the wrong branch.
    """
    senses = {n: _is_maximize(n) for n in AVAILABLE}
    maximizing = sorted(n for n, mx in senses.items() if mx)
    print(
        f"[sense probe] {len(senses)} instances read: "
        f"{len(maximizing)} maximize {maximizing}, "
        f"{len(senses) - len(maximizing)} minimize"
    )
    assert len(senses) == len(AVAILABLE), "an instance did not yield a sense"


class TestKnownOptima:
    """Verify solver finds correct global optima on known instances."""

    @pytest.mark.parametrize("instance,expected", list(KNOWN_OPTIMA.items()))
    def test_optimal_value(self, instance: str, expected: float):
        """Solver objective must match known optimum within tolerance."""
        sol = solve_instance(instance)

        if sol.status != "optimal":
            pytest.skip(f"Not solved to optimality (status={sol.status})")

        assert sol.objective is not None, "Optimal status but no objective"
        diff = abs(sol.objective - expected)
        tol = ABS_TOL + REL_TOL * abs(expected)
        assert diff <= tol, (
            f"INCORRECT: {instance} obj={sol.objective:.8e} "
            f"expected={expected:.8e} diff={diff:.2e} tol={tol:.2e}"
        )

    @pytest.mark.parametrize("instance,expected", list(KNOWN_OPTIMA.items()))
    def test_bound_validity(self, instance: str, expected: float):
        """The dual bound must never cut off the true optimum.

        Which direction that is depends on the objective sense, and three of the
        instances in ``KNOWN_OPTIMA`` (the ``procurement*`` family) maximize. An
        assertion hardcoded to minimization is wrong twice over on those: it
        fires on a perfectly valid upper bound, and it can never catch a genuinely
        invalid one, because it is testing the wrong inequality. The sense is read
        from the parsed model rather than assumed.
        """
        sol = solve_instance(instance)

        if sol.bound is None:
            pytest.skip("No bound reported")

        maximizing = _is_maximize(instance)
        relation = "bound below optimum" if maximizing else "bound exceeds optimum"

        assert bound_is_valid(sol.bound, expected, maximize=maximizing), (
            f"INVALID BOUND: {instance} ({'max' if maximizing else 'min'}) "
            f"bound={sol.bound:.8e} optimal={expected:.8e} ({relation}!)"
        )


# ─────────────────────────────────────────────────────────────
# 2. FEASIBILITY VERIFICATION
# ─────────────────────────────────────────────────────────────


class TestFeasibility:
    """Verify that reported solutions actually satisfy constraints."""

    FEASIBILITY_TOL = 1e-6

    @pytest.mark.parametrize("instance", list(KNOWN_OPTIMA.keys())[:10])
    def test_solution_feasibility(self, instance: str):
        """Returned solution point must satisfy all constraints."""
        sol = solve_instance(instance)

        if not sol.x or sol.status not in ("optimal", "feasible"):
            pytest.skip("No feasible solution")

        x_flat = _flat_x(instance, sol)
        assert x_flat is not None, "solution reported but could not be flattened"

        from discopt.warm_start import check_feasibility

        # The reported point is checked against the model the solver was given,
        # not against the solver's own internal view of it — the point of the
        # test is to catch a solver that believes an infeasible point.
        ok, violations = check_feasibility(_load(instance), x_flat, tol=self.FEASIBILITY_TOL)
        assert ok, f"INFEASIBLE incumbent for {instance}: " + "; ".join(violations[:5])

    @pytest.mark.parametrize("instance", list(KNOWN_OPTIMA.keys())[:10])
    def test_integrality(self, instance: str):
        """Integer variables must have integer values in solution."""
        sol = solve_instance(instance)

        if not sol.x or sol.status not in ("optimal", "feasible"):
            pytest.skip("No feasible solution")

        model = _load(instance)
        checked = 0
        for var in model._variables:
            if var.var_type.value not in ("binary", "integer"):
                continue
            for value in np.asarray(sol.x[var.name], dtype=float).ravel():
                checked += 1
                assert abs(value - round(value)) < 1e-5, (
                    f"{instance}: {var.name} = {value!r} is not integral"
                )
        if checked == 0:
            pytest.skip(f"{instance} has no discrete variables")


# ─────────────────────────────────────────────────────────────
# 3. DETERMINISM
# ─────────────────────────────────────────────────────────────


class TestDeterminism:
    """Verify solver produces identical results across runs."""

    NUM_RUNS = 3
    # Prefer instances that actually exist here: on CI only the five in-repo
    # `.nl` files resolve, and slicing the raw dict would spend two of the five
    # parametrisations on instances that can only skip.
    INSTANCES = AVAILABLE[:5]

    @pytest.mark.parametrize("instance", INSTANCES)
    def test_deterministic_objective(self, instance: str):
        """Objective value must be identical across runs."""
        objectives = []
        for _ in range(self.NUM_RUNS):
            # cache=False: a memoised solve would return one object to all three
            # repetitions and this test would pass without the solver ever having
            # run twice.
            sol = solve_instance(instance, cache=False, deterministic=True)
            if sol.objective is not None:
                objectives.append(sol.objective)

        if len(objectives) < 2:
            pytest.skip("Not enough solved runs")

        for i in range(1, len(objectives)):
            assert abs(objectives[i] - objectives[0]) < 1e-10, (
                f"Non-deterministic: run 0 obj={objectives[0]:.10e} "
                f"run {i} obj={objectives[i]:.10e}"
            )

    @pytest.mark.parametrize("instance", INSTANCES)
    def test_deterministic_node_count(self, instance: str):
        """Node count must be identical in deterministic mode."""
        node_counts = []
        for _ in range(self.NUM_RUNS):
            sol = solve_instance(instance, cache=False, deterministic=True)
            node_counts.append(sol.node_count)

        if len(node_counts) < 2:
            pytest.skip("Not enough runs")

        assert all(n == node_counts[0] for n in node_counts), (
            f"Non-deterministic node counts: {node_counts}"
        )


# ─────────────────────────────────────────────────────────────
# 4. EDGE CASES
# ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test solver behavior on degenerate and boundary cases.

    These build their models directly rather than reading a corpus, so they run
    everywhere — including CI, which ships no MINLPLib snapshot.
    """

    @staticmethod
    def _model(name: str):
        from discopt import Model

        return Model(name)

    def test_infeasible_detection(self):
        """Solver must correctly report infeasibility."""
        m = self._model("infeasible")
        x = m.continuous("x", lb=0, ub=1)
        m.subject_to(x >= 2)
        m.minimize(x)
        assert m.solve(time_limit=TIME_LIMIT).status == "infeasible"

    def test_unbounded_detection(self):
        """Solver must correctly report unboundedness."""
        m = self._model("unbounded")
        # 1e20 is the LP layer's INF sentinel, so this is an unbounded bound and
        # not merely a very wide one.
        x = m.continuous("x", lb=-1e20, ub=1e20)
        m.minimize(x)
        assert m.solve(time_limit=TIME_LIMIT).status == "unbounded"

    def test_fixed_variables(self):
        """Handle variables with lb == ub."""
        m = self._model("fixed")
        x = m.continuous("x", lb=2.0, ub=2.0)
        y = m.continuous("y", lb=0.0, ub=10.0)
        m.subject_to(y >= x)
        m.minimize(x + y)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(4.0, abs=1e-5)

    def test_empty_problem(self):
        """Handle problem with no constraints."""
        m = self._model("unconstrained")
        x = m.continuous("x", lb=-3.0, ub=5.0)
        m.minimize(x * x)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(0.0, abs=1e-5)

    def test_single_variable(self):
        """Handle trivial single-variable problems."""
        m = self._model("single")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        m.minimize((x - 1.25) ** 2)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(0.0, abs=1e-5)
        assert float(np.ravel(res.x["x"])[0]) == pytest.approx(1.25, abs=1e-4)

    def test_all_integer(self):
        """Handle pure integer (no continuous) problems."""
        m = self._model("all_integer")
        a = m.integer("a", lb=0, ub=10)
        b = m.integer("b", lb=0, ub=10)
        m.subject_to(a + b >= 7)
        m.minimize(3 * a + 2 * b)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(14.0, abs=1e-4)

    def test_all_continuous(self):
        """Handle pure NLP (no integer) problems."""
        m = self._model("all_continuous")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        m.subject_to(x + y >= 1)
        m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(0.0, abs=1e-5)

    def test_linear_constraints_only(self):
        """Handle problems with only linear constraints."""
        m = self._model("linear_only")
        x = m.continuous("x", lb=0.0, ub=10.0)
        y = m.binary("y")
        m.subject_to(x <= 10 * y)
        m.subject_to(x >= 3)
        m.minimize(x + 5 * y)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(8.0, abs=1e-4)

    def test_very_tight_bounds(self):
        """Handle near-fixed variables (ub - lb < 1e-8)."""
        m = self._model("tight_bounds")
        x = m.continuous("x", lb=1.0, ub=1.0 + 1e-9)
        y = m.continuous("y", lb=0.0, ub=5.0)
        m.subject_to(y >= 2 * x)
        m.minimize(y)
        res = m.solve(time_limit=TIME_LIMIT)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(2.0, abs=1e-5)

    def test_large_coefficient_range(self):
        """Handle poorly scaled problems (coefficient range > 1e8)."""
        m = self._model("badly_scaled")
        x = m.continuous("x", lb=0.0, ub=1e6)
        y = m.continuous("y", lb=0.0, ub=1e-3)
        m.subject_to(1e8 * y + x >= 1e5)
        m.minimize(x + 1e9 * y)
        res = m.solve(time_limit=TIME_LIMIT)
        # Buying the constraint through x costs 1 per unit and through y costs
        # 10, so the optimum is all-x at 1e5 -- a solver that lets the 1e9
        # coefficient dominate its scaling would take the y route.
        assert res.status == "optimal"
        assert res.objective == pytest.approx(1e5, rel=1e-6)
