"""Exact reduction as the pre-pass of the presolve-free confirmation route
(``discopt._iis_reduce``).

On MIPcc27's general-integer models presolve-free HiGHS could not confirm the
subsets the deletion filter passes through (undecided at 60-300 s) while
presolve-on proved them at once. The engine now reduces a set exactly -- five
textbook rules over rationals, each preserving "feasible set nonempty" in both
directions -- and runs presolve-free HiGHS on the reduced model: dijle's path
sets in 1-11 s, dingo's in 0.3 s, dives' in 0.0 s.

The rules can only ever be trusted if they are sound, so the first test is a
differential one against HiGHS on random models, with infeasible ones in the mix.
"""

import itertools
import random
from fractions import Fraction

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_backend as _bk
from discopt import _iis_milp
from discopt._iis_milp import milp_iis
from discopt._iis_reduce import Infeasible, Model, reduced_infeasible

highspy = pytest.importorskip("highspy")


def _highs(A, rlo, rhi, lb, ub, held, presolve):
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", presolve)
    h.setOptionValue("time_limit", 20.0)
    n = A.shape[1]
    inf = highspy.kHighsInf
    h.addVars(n, np.where(np.isfinite(lb), lb, -inf), np.where(np.isfinite(ub), ub, inf))
    vt = highspy.HighsVarType
    h.changeColsIntegrality(
        n,
        np.arange(n, dtype=np.int32),
        np.array([vt.kInteger if k else vt.kContinuous for k in held]),
    )
    A = A.tocsr()
    for r in range(A.shape[0]):
        s, e = A.indptr[r], A.indptr[r + 1]
        h.addRow(
            rlo[r] if np.isfinite(rlo[r]) else -inf,
            rhi[r] if np.isfinite(rhi[r]) else inf,
            e - s,
            A.indices[s:e].astype(np.int32),
            A.data[s:e],
        )
    h.run()
    return h.getModelStatus().name


def _random_model(rng):
    n, m = rng.randint(2, 7), rng.randint(1, 7)
    A = np.zeros((m, n))
    for r in range(m):
        for j in rng.sample(range(n), rng.randint(1, min(n, 3))):
            A[r, j] = rng.choice([-172, -3, -2, -1, 1, 1, 2, 3, 5, 9])
    rlo = np.array([rng.choice([-np.inf, rng.randint(-6, 6)]) for _ in range(m)], float)
    rhi = np.array([rng.choice([np.inf, rng.randint(-6, 8)]) for _ in range(m)], float)
    for r in range(m):
        if rng.random() < 0.15:
            rhi[r] = rlo[r] = rng.randint(-3, 3)
    lb = np.array([rng.choice([0, 0, -2, -np.inf]) for _ in range(n)], float)
    ub = np.array([rng.choice([1, 1, 3, 7, np.inf]) for _ in range(n)], float)
    held = np.array([rng.random() < 0.7 for _ in range(n)])
    return sp.csr_matrix(A), rlo, rhi, lb, ub, held


@pytest.mark.smoke
def test_reduction_preserves_feasibility_both_ways_on_random_models():
    rng = random.Random(11)
    decided = {"kInfeasible": False, "kOptimal": True}
    checked = infeasible = by_reduction = 0
    for _ in range(150):
        A, rlo, rhi, lb, ub, held = _random_model(rng)
        on = _highs(A, rlo, rhi, lb, ub, held, "on")
        if on not in decided or _highs(A, rlo, rhi, lb, ub, held, "off") != on:
            continue
        # The route's own verdict: True (infeasible) or None (no conclusion).
        verdict = reduced_infeasible(A, rlo, rhi, lb, ub, held, None)
        if verdict is True:
            assert not decided[on], (A.toarray(), rlo, rhi, lb, ub, held)  # never on a feasible set
            by_reduction += 1
        # The reduction itself, both ways, where it is exported to HiGHS.
        model = Model(A, rlo, rhi, lb, ub, held)
        try:
            assert model.reduce() == "fixpoint"
            red = model.export()
            mags = [np.abs(red[0].data)] + [np.abs(v[np.isfinite(v)]) for v in red[1:5]]
            if any(m.size and m.max() > 1e9 for m in mags):
                continue  # the route declines these (no conclusion), see above
            if red[0].shape[0] == 0:
                red_feasible = True
            else:
                st = _highs(*red, "on")
                assert st in decided, st
                red_feasible = decided[st]
        except Infeasible:
            red_feasible = False
        assert red_feasible == decided[on], (A.toarray(), rlo, rhi, lb, ub, held)
        checked += 1
        infeasible += not decided[on]
    assert checked >= 140 and infeasible >= 40 and by_reduction >= 30


def test_coefficient_tightening_and_aggregation_reach_the_small_model():
    """dijle's shape: u - 172 x <= 0 with u <= 1 is u <= x (tightening); u is
    wanted up by its other row and capped only here, so u := x (aggregation)."""
    # cols: x (binary), u in [0,1] continuous, y binary
    A = sp.csr_matrix(np.array([[-172.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]))
    rlo = np.array([-np.inf, 2.0, -np.inf])  # row1: u + y >= 2
    rhi = np.array([0.0, np.inf, 1.0])  # row0: u <= 172 x ; row2: x + y <= 1
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    held = np.array([True, False, True])
    model = Model(A, rlo, rhi, lb, ub, held)
    with pytest.raises(Infeasible):  # u + y >= 2 forces u = 1 = x and y = 1: x + y = 2 > 1
        model.reduce()
    assert model.n_rules["P"] >= 1


def test_aggregation_substitutes_and_keeps_the_answer():
    m = Model(
        sp.csr_matrix(np.array([[1.0, -1.0], [1.0, 0.0]])),
        np.array([-np.inf, 1.0]),  # row1: u >= 1 (wants u up)
        np.array([0.0, np.inf]),  # row0: u - x <= 0 (caps u by x)
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([False, True]),
    )
    assert m.reduce() == "fixpoint"
    assert m.lb[1] == Fraction(1)  # x >= 1 after u := x or propagation


def test_reduced_route_never_concludes_on_a_feasible_set():
    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, -1.0]]))
    rlo, rhi = np.array([1.0, -np.inf]), np.array([np.inf, 0.0])
    lb, ub = np.zeros(2), np.array([3.0, 3.0])
    assert reduced_infeasible(A, rlo, rhi, lb, ub, np.array([True, True]), None) is None


def _dijle_like(n_rows=40):
    """Knapsack-coupled big-M rows over binaries and bounded integers: the
    general-integer pattern, infeasible through integrality."""
    m = dm.Model("g")
    x = m.binary("x", shape=(n_rows,))
    u = m.integer("u", shape=(n_rows,), lb=0, ub=172)
    for k in range(n_rows):
        m.subject_to(u[k] - 172 * x[k] <= 0, name=f"link{k}")
        m.subject_to(u[k] >= 2, name=f"dem{k}")
    m.subject_to(sum(x[k] for k in range(n_rows)) <= n_rows - 1, name="cap")
    m.minimize(0 * x[0])
    return m


def test_engine_confirms_through_the_reduction_when_presolve_free_cannot(monkeypatch):
    real = _iis_milp._Engine._solve

    def solve(self, presolve, stop_at=None):
        if presolve == "off":
            return _bk.OTHER  # the raw route fails
        return real(self, presolve, stop_at)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    res = milp_iis(_dijle_like(), total_time_limit=60.0, sat_candidate=False)
    assert res.n_reduced_confirmed >= 1
    assert res.complete and not res.fallback


def test_the_route_skips_the_solve_when_the_reduction_did_little(monkeypatch):
    """#2145 review: an undecided solve of a barely-reduced model ate the whole
    confirmation budget, starving the raw presolve-free route and the rational
    decider. When the reduction keeps more than 80% of the rows the route draws
    no conclusion at once and leaves the budget to them."""
    import time

    calls = {"n": 0}
    real = highspy.Highs.run

    def counting_run(self):
        calls["n"] += 1
        return real(self)

    monkeypatch.setattr(highspy.Highs, "run", counting_run)
    # Rows the five rules cannot touch: two-variable rows over wide integers.
    n = 6
    A = sp.csr_matrix(
        np.array([[1.0 if c in (r, (r + 1) % n) else 0.0 for c in range(n)] for r in range(n)])
    )
    rlo, rhi = np.full(n, 1.0), np.full(n, 1.0)  # x_r + x_{r+1} = 1 around an even cycle
    lb, ub = np.full(n, -50.0), np.full(n, 50.0)
    t = time.perf_counter()
    assert reduced_infeasible(A, rlo, rhi, lb, ub, np.ones(n, bool), t + 30.0) is None
    assert calls["n"] == 0  # no HiGHS solve of a model the reduction did not shrink


def test_a_refuted_presolve_on_claim_is_not_recorded_as_proven(monkeypatch):
    """#2145 review: probe() recorded a presolve-on Infeasible before the
    confirm_each cross-check; a claim the presolve-free route refutes must not
    stay on record (it would let a later confirmation skip presolve-on)."""
    eng = _iis_milp._Engine(
        sp.csr_matrix(np.array([[1.0, 1.0]])),
        np.array([1.0]),
        np.array([1e20]),
        np.zeros(2),
        np.ones(2),
        np.array([True, True]),
        {0: np.array([0])},
        30.0,
        confirm_each=True,
    )
    real = _iis_milp._Engine._solve

    def solve(self, presolve, stop_at=None):
        if presolve == "choose":
            return _bk.INFEASIBLE  # a false presolve-on claim
        return real(self, presolve, stop_at)

    monkeypatch.setattr(_iis_milp._Engine, "_solve", solve)
    res, x = eng.probe({0}, set(), {0, 1})  # x0 + x1 >= 1: feasible in truth
    assert res is False and x is not None
    assert not eng.proven_on({0}, set(), {0, 1})


def _decimal_knapsack(rhs, card):
    """0.3 a + 0.5 b + 0.7 c + 0.05 d = rhs and a + b + c + d = card over
    integers in [0, 3], plus four rows the reduction drops (e_k <= 2 on e_k in
    [0, 1]) so the export route runs: propagation alone decides neither case."""
    A = np.zeros((6, 8))
    A[0, :4] = [0.3, 0.5, 0.7, 0.05]
    A[1, :4] = 1.0
    for k in range(4):
        A[2 + k, 4 + k] = 1.0
    rlo = np.array([rhs, card] + [-np.inf] * 4)
    rhi = np.array([rhs, card] + [2.0] * 4)
    ub = np.array([3.0] * 4 + [1.0] * 4)
    return sp.csr_matrix(A), rlo, rhi, np.zeros(8), ub, np.array([True] * 4 + [False] * 4)


def test_decimal_row_reaches_the_reduced_solve_and_concludes_correctly():
    """The row [0.3, 0.5, 0.7, 0.05] used to be scaled by the lcm of its binary
    denominators (2**56), about 2e16 per coefficient, past _MAX_MAGNITUDE: the
    route drew no conclusion on any decimal row. Read as decimals it scales by
    20 to [6, 10, 14, 1]. Activity 1.0 with 3 items is infeasible; with 2 items
    (0.3 + 0.7) it is feasible and must never be concluded infeasible."""
    w = [6, 10, 14, 1]  # the oracle, by enumeration in units of 0.05

    def attainable(card):
        return any(
            sum(c * k for c, k in zip(w, x)) == 20 and sum(x) == card
            for x in itertools.product(range(4), repeat=4)
        )

    assert not attainable(3) and attainable(2)
    model = Model(*_decimal_knapsack(1.0, 3))
    assert model.reduce() == "fixpoint"
    Ar = model.export()[0]
    assert Ar.shape[0] == 2 and sorted(set(Ar.data.tolist())) == [1.0, 6.0, 10.0, 14.0]
    for engine in ("highs", "rust"):
        assert reduced_infeasible(*_decimal_knapsack(1.0, 3), None, engine=engine) is True
        assert reduced_infeasible(*_decimal_knapsack(1.0, 2), None, engine=engine) is None
    assert _highs(*_decimal_knapsack(1.0, 3), "off") == "kInfeasible"
    assert _highs(*_decimal_knapsack(1.0, 2), "off") == "kOptimal"


def test_a_feasible_decimal_row_is_not_proven_infeasible_by_binary_rounding():
    """0.3 a + 0.7 c = 1 holds at a = c = 1 (HiGHS agrees). Over the floats'
    binary values the activity is 1 - 2**-54, and the exact reduction used to
    return True -- an infeasibility "proof" of a feasible set."""
    A = sp.csr_matrix(np.array([[0.3, 0.7]]))
    args = (A, np.array([1.0]), np.array([1.0]), np.zeros(2), np.ones(2), np.array([True, True]))
    assert _highs(*args, "off") == "kOptimal"
    assert reduced_infeasible(*args, None) is None


def test_decimal_models_never_concluded_infeasible_when_feasible():
    """Differential: random decimal-coefficient models; every True must be an
    infeasible model under both HiGHS presolve settings."""
    rng = random.Random(5)
    coefs = [-0.7, -0.35, -0.2, 0.05, 0.1, 0.3, 0.5, 0.7, 1.25]
    n_true = n_checked = 0
    for _ in range(200):
        m, n = rng.randint(1, 4), rng.randint(1, 4)
        A = np.zeros((m, n))
        for r in range(m):
            for j in rng.sample(range(n), rng.randint(1, n)):
                A[r, j] = rng.choice(coefs)
        side = [round(rng.uniform(-1.5, 1.5), 2) for _ in range(m)]
        kinds = [rng.choice(["eq", "lo", "hi"]) for _ in range(m)]
        rlo = np.array([s if k in ("eq", "lo") else -np.inf for s, k in zip(side, kinds)])
        rhi = np.array([s if k in ("eq", "hi") else np.inf for s, k in zip(side, kinds)])
        lb, ub = np.zeros(n), np.array([rng.choice([1.0, 2.0, 3.0]) for _ in range(n)])
        held = np.array([rng.random() < 0.8 for _ in range(n)])
        args = (sp.csr_matrix(A), rlo, rhi, lb, ub, held)
        on, off = _highs(*args, "on"), _highs(*args, "off")
        if on not in ("kOptimal", "kInfeasible") or off != on:
            continue
        n_checked += 1
        if reduced_infeasible(*args, None, engine="highs") is True:
            n_true += 1
            assert on == "kInfeasible", args
    assert n_checked >= 150 and n_true >= 20, (n_checked, n_true)
