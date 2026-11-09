"""The CDCL core candidate for the MILP IIS engine (``_iis_milp._binary_core``) on clause models.

On a pure-binary clause model the engine starts from a small unsatisfiable row set
found by discopt's own CDCL solver (``discopt._rust.PySatSolver``) instead of
discovering it with thousands of MILP probes. Measured on MIPcc27's smallset:
bf2670-001 went from 399 elements / 1,913 s to 145 / 6.4 s (87 conflicts).

The candidate is only a proposal. These tests pin that: results stay
independently infeasible and certified; a model the generator does not apply to
is computed exactly as without it; and a wrong candidate is rejected by the
HiGHS base proof, after which the full model is used.
"""

import discopt.modeling as dm
import numpy as np
import pytest

highspy = pytest.importorskip("highspy")
from discopt._iis_milp import milp_iis  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_highs_engine(monkeypatch):
    """These tests' budgets and probe counts were tuned on the HiGHS IIS engine.
    Since highs-parity P6.4 (2026-11-07) the default engine is the in-house
    one, so these HiGHS-path tests pin ``DISCOPT_IIS_ENGINE=highs`` (a test that sets it still
    overrides)."""
    monkeypatch.setenv("DISCOPT_IIS_ENGINE", "highs")


def _clause_model(rng, n, m, k=3):
    """Random k-clause model over binaries: clause -> sum of literals >= 1."""
    mod = dm.Model("cnf")
    x = [mod.binary(f"x{j}") for j in range(n)]
    cls = []
    for _ in range(m):
        # distinct variables: a repeated one would give a 2 or 0 coefficient,
        # which is (correctly) not a +-1 clause row
        lits = [(int(j), bool(rng.integers(2))) for j in rng.choice(n, size=k, replace=False)]
        cls.append(lits)
        # positive literal x_j, negative (1 - x_j): sum >= 1  <=>  +-x terms >= 1 - #neg
        expr = sum((-x[j] if neg else x[j]) for j, neg in lits)
        mod.subject_to(expr >= 1 - sum(neg for _, neg in lits))
    mod.minimize(0 * x[0])
    return mod, cls


def _infeasible_with_highs(n, cls, rows, ints):
    """Independent check: the kept clause rows with the kept integralities."""
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.silent()
    h.setOptionValue("presolve", "off")
    held = set(ints)
    h.addVars(
        n,
        np.array([0.0 if j in held else -inf for j in range(n)]),
        np.array([1.0 if j in held else inf for j in range(n)]),
    )
    for r in rows:
        lits = cls[r]
        cols = np.array([j for j, _ in lits], dtype=np.int32)
        vals = np.array([-1.0 if neg else 1.0 for _, neg in lits])
        h.addRow(1 - sum(neg for _, neg in lits), inf, len(cols), cols, vals)
    if held:
        idx = np.array(sorted(held), dtype=np.int32)
        h.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx)))
    h.run()
    return h.getModelStatus() == highspy.HighsModelStatus.kInfeasible


def _is_unsat(n, cls):
    return not any(
        all(any(((bits >> j) & 1) != neg for j, neg in c) for c in cls) for bits in range(1 << n)
    )


@pytest.mark.smoke
def test_clause_candidate_results_are_infeasible_and_certified():
    rng = np.random.default_rng(1)
    used = checked = 0
    for _ in range(80):
        n, m = 8, int(rng.integers(30, 50))
        mod, cls = _clause_model(rng, n, m)
        if not _is_unsat(n, cls):
            continue
        res = milp_iis(mod, time_limit=10)
        used += res.sat_rows is not None
        assert res.proven_irreducible and res.complete
        assert _infeasible_with_highs(n, cls, res.constraints, res.integrality)
        size = len(res.constraints) + len(res.bounds) + len(res.integrality)
        assert len(res.certificates) == size  # fully certified
        checked += 1
    print(f"[sat-candidate] models={checked} candidate_used={used}")
    assert checked >= 10 and used >= 10, (checked, used)


@pytest.mark.smoke
def test_non_clause_model_is_computed_exactly_as_without_the_candidate():
    m = dm.Model("p")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    # 3x = 1: integer-infeasible, but not by parity (2x = 1 would be a mod-2
    # certificate, which the parity candidate legitimately takes).
    m.subject_to(3 * x == 1, name="half")
    m.subject_to(y <= 3, name="red_herring")
    m.minimize(0 * x)
    a = milp_iis(m)
    b = milp_iis(m, sat_candidate=False)
    assert a.sat_rows is None
    assert (a.constraints, a.bounds, a.integrality, a.n_solves) == (
        b.constraints,
        b.bounds,
        b.integrality,
        b.n_solves,
    )


@pytest.mark.smoke
def test_a_wrong_candidate_is_rejected_by_the_highs_base_proof(monkeypatch):
    from discopt import _iis_milp

    rng = np.random.default_rng(5)
    while True:
        mod, cls = _clause_model(rng, 8, 40)
        if _is_unsat(8, cls):
            break
    # Propose a single row: certainly satisfiable, so HiGHS must refute it.
    monkeypatch.setattr(_iis_milp, "_binary_core", lambda *a, **k: ({0}, 0))
    res = milp_iis(mod, time_limit=10)
    assert res.sat_rows is None  # candidate discarded
    assert res.proven_irreducible
    assert _infeasible_with_highs(8, cls, res.constraints, res.integrality)


@pytest.mark.smoke
def test_sat_solver_binding_cores():
    from discopt._rust import PySatSolver

    s = PySatSolver(3)
    s.add_clause([1, 2])
    s.add_clause([-1, 3])
    s.add_clause([-2, 3])
    assert s.solve([]) is True
    x = s.model()
    assert (x[0] or x[1]) and (not x[0] or x[2]) and (not x[1] or x[2])
    assert s.solve([-3]) is False
    assert s.core() == [-3]
    with pytest.raises(ValueError):
        s.add_clause([4])


def _pigeonhole_model(p, h):
    """PHP(p, h) as a clause model: every pigeon in a hole, no hole shared."""
    mod = dm.Model("php")
    x = mod.binary("x", shape=(p, h))
    for i in range(p):
        mod.subject_to(sum(x[i, j] for j in range(h)) >= 1)
    for j in range(h):
        for a in range(p):
            for b in range(a + 1, p):
                mod.subject_to(-x[a, j] - x[b, j] >= -1)  # not both: clause form
    mod.minimize(0 * x[0, 0])
    return mod


@pytest.mark.smoke
def test_candidate_stage_respects_the_time_budget():
    """Pigeonhole is hard for CDCL: PHP(10,9) ran > 280 s inside the 200k-conflict
    budget, and on smallset's hole9/hole10 the candidate stage overran a 60 s
    total_time_limit until the process was killed. The stage is now sliced and
    bounded by a conflict budget and a share of the time budget; the engine then
    proceeds without it."""
    import time

    mod = _pigeonhole_model(10, 9)
    budget = 6.0
    t0 = time.perf_counter()
    res = milp_iis(mod, time_limit=10, total_time_limit=budget)
    elapsed = time.perf_counter() - t0
    assert elapsed < budget + 4.0, elapsed  # CI-safe slack; ~budget locally
    assert res.sat_rows is None  # no candidate in time: engine ran without one
    assert res.sat_conflicts > 0  # the SAT stage did run (and was cut off)
