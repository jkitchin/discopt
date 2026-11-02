"""``compute_iis(include_integrality=True)``: integrality as an IIS element.

For a MIP the integrality of a column is a removable member of an infeasible
subsystem (Guieu & Chinneck 1999), and usually the one that dominates its size:
on MIPcc27's smallset the rows discopt kept matched Gurobi's reference exactly
while every integrality was kept (aim-50: 50 integralities vs Gurobi's 8).

Semantics pinned here (also the MIPcc27 solution format): the bounds of a column
whose integrality is kept always apply and are not members; once integrality is
relaxed its bounds are ordinary members. Every kept member carries a certificate
-- a point satisfying all other members and violating it -- checked here
independently of the engine, against a HiGHS model built from the generating
matrix rather than discopt's own extraction.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.infeasibility import compute_iis

highspy = pytest.importorskip("highspy")


@pytest.mark.smoke
def test_solution_format_example():
    """The MIPcc27 SOLUTION_FORMAT.md example: IIS {c1, c2}, integrality relaxed."""
    m = dm.Model("ex")
    x = m.integer("x", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] + x[1] <= 1.5, name="c1")
    m.subject_to(x[0] + x[1] >= 2.5, name="c2")
    m.subject_to(x[0] - x[1] <= 0, name="c3")
    m.minimize(0 * x[0])
    iis = compute_iis(m, include_integrality=True)
    assert {c.name for c in iis.constraints} == {"c1", "c2"}
    assert iis.integrality_elements == []
    assert iis.bound_elements == []
    assert iis.proven_irreducible
    assert set(iis.certificates) == {("constraint", 0), ("constraint", 1)}
    assert len(iis) == 2


@pytest.mark.smoke
def test_lp_feasible_mip_infeasible_keeps_only_needed_integrality():
    m = dm.Model("p")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(2 * x == 1, name="half")
    m.subject_to(y <= 3, name="red_herring")
    m.minimize(0 * x)
    iis = compute_iis(m, include_integrality=True)
    assert [c.name for c in iis.constraints] == ["half"]
    assert [(v.name, idx) for v, idx in iis.integrality_elements] == [("x", ())]
    cert = iis.certificates[("integrality", 0)]
    assert abs(cert[0] - round(cert[0])) > 1e-6  # fractional in x: proves it
    assert 2 * cert[0] == pytest.approx(1.0)


@pytest.mark.smoke
def test_relaxed_binary_bounds_become_members():
    """z binary, z >= 2: the conflict is {z >= 2, z <= 1}, integrality not needed."""
    m = dm.Model("b")
    z = m.binary("z")
    m.subject_to(z >= 2, name="too_big")
    m.minimize(0 * z)
    iis = compute_iis(m, include_integrality=True)
    assert [c.name for c in iis.constraints] == ["too_big"]
    assert iis.integrality_elements == []
    assert [(v.name, s) for v, s, _ in iis.bound_elements] == [("z", "upper")]


@pytest.mark.smoke
def test_feasible_model_raises():
    m = dm.Model("f")
    x = m.integer("x", lb=0, ub=5)
    m.subject_to(x >= 1)
    m.minimize(x)
    with pytest.raises(ValueError, match="feasible"):
        compute_iis(m, include_integrality=True)


@pytest.mark.smoke
def test_nonlinear_model_refused():
    m = dm.Model("nl")
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(x * x >= 4)
    m.minimize(x)
    with pytest.raises(NotImplementedError):
        compute_iis(m, include_integrality=True)


def _independent_check(
    A, rlo, rhi, lb, ub, is_int, kept_rows, kept_lb, kept_ub, kept_int, x, drop=None, tol=1e-6
):
    """Arithmetic check of a certificate against the generating data."""
    held = set(kept_int)
    rows = [r for r in kept_rows if drop != ("constraint", r)]
    act = A[rows] @ x
    ok = np.all(act >= rlo[rows] - tol) and np.all(act <= rhi[rows] + tol)
    for j in range(len(lb)):
        apply_lb = j in held or j in kept_lb or drop == ("integrality", j)
        apply_ub = j in held or j in kept_ub or drop == ("integrality", j)
        if drop == ("bound", j, "lower"):
            apply_lb = False
        if drop == ("bound", j, "upper"):
            apply_ub = False
        ok &= (not apply_lb) or x[j] >= lb[j] - tol
        ok &= (not apply_ub) or x[j] <= ub[j] + tol
    for j in held:
        if drop == ("integrality", j):
            continue
        ok &= abs(x[j] - round(x[j])) <= tol
    return bool(ok)


def _subsystem_infeasible(A, rlo, rhi, lb, ub, kept_rows, kept_lb, kept_ub, kept_int):
    inf = highspy.kHighsInf
    n = A.shape[1]
    h = highspy.Highs()
    h.silent()
    held = set(kept_int)
    clb = [lb[j] if (j in held or j in kept_lb) else -inf for j in range(n)]
    cub = [ub[j] if (j in held or j in kept_ub) else inf for j in range(n)]
    h.addVars(n, np.array(clb), np.array(cub))
    for r in kept_rows:
        nz = np.flatnonzero(A[r])
        h.addRow(rlo[r], rhi[r], len(nz), nz.astype(np.int32), A[r, nz])
    if held:
        idx = np.array(sorted(held), dtype=np.int32)
        h.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx)))
    h.run()
    return h.getModelStatus() == highspy.HighsModelStatus.kInfeasible


@pytest.mark.smoke
def test_random_milps_iis_is_infeasible_and_certified():
    """Property test over random infeasible MILPs (seeded), checked independently."""
    rng = np.random.default_rng(1873)
    n_checked = 0
    n_models = 0
    for _ in range(60):
        n, mrows = 4, 5
        A = rng.integers(-3, 4, size=(mrows, n)).astype(float)
        rhs = rng.integers(-4, 5, size=mrows).astype(float) + rng.choice([0.0, 0.5], mrows)
        lb, ub = np.zeros(n), np.full(n, 3.0)
        is_int = rng.random(n) < 0.7
        m = dm.Model("r")
        xi = [
            (m.integer(f"x{j}", lb=0, ub=3) if is_int[j] else m.continuous(f"x{j}", lb=0, ub=3))
            for j in range(n)
        ]
        for r in range(mrows):
            m.subject_to(sum(A[r, j] * xi[j] for j in range(n)) <= rhs[r], name=f"r{r}")
        m.minimize(0 * xi[0])
        rlo, rhi = np.full(mrows, -np.inf), rhs.copy()
        if not _subsystem_infeasible(
            A, rlo, rhi, lb, ub, range(mrows), set(range(n)), set(range(n)), np.flatnonzero(is_int)
        ):
            continue
        n_models += 1
        iis = compute_iis(m, include_integrality=True, time_limit=10)
        assert iis.proven_irreducible
        kept_rows = [int(c.name[1:]) for c in iis.constraints]
        kept_lb = {int(v.name[1:]) for v, s, _ in iis.bound_elements if s == "lower"}
        kept_ub = {int(v.name[1:]) for v, s, _ in iis.bound_elements if s == "upper"}
        kept_int = [int(v.name[1:]) for v, _ in iis.integrality_elements]
        assert _subsystem_infeasible(A, rlo, rhi, lb, ub, kept_rows, kept_lb, kept_ub, kept_int)
        members = (
            [("constraint", r) for r in kept_rows]
            + [("bound", j, "lower") for j in kept_lb]
            + [("bound", j, "upper") for j in kept_ub]
            + [("integrality", j) for j in kept_int]
        )
        # Here constraint index == row index and flat column == j.
        for key in members:
            assert key in iis.certificates, key
            x = iis.certificates[key]
            assert _independent_check(
                A, rlo, rhi, lb, ub, is_int, kept_rows, kept_lb, kept_ub, kept_int, x, drop=key
            )
            assert not _independent_check(
                A, rlo, rhi, lb, ub, is_int, kept_rows, kept_lb, kept_ub, kept_int, x
            )
            n_checked += 1
    print(f"[iis-property] models={n_models} certificates_checked={n_checked}")
    assert n_models >= 10, n_models
    assert n_checked >= 30, n_checked


@pytest.mark.smoke
def test_every_infeasible_verdict_is_cross_checked_without_presolve(monkeypatch):
    """#1634: a presolve-on ``Infeasible`` drops an element only if presolve-off agrees."""
    from discopt._iis_milp import milp_iis

    log = []
    real_run = highspy.Highs.run

    def run(self):
        out = real_run(self)
        pre = self.getOptionValue("presolve")
        pre = pre[1] if isinstance(pre, tuple) else pre  # (status, value) in some highspy
        log.append((pre, self.getModelStatus()))
        return out

    monkeypatch.setattr(highspy.Highs, "run", run)
    m = dm.Model("ex")
    x = m.integer("x", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] + x[1] <= 1.5, name="c1")
    m.subject_to(x[0] + x[1] >= 2.5, name="c2")
    m.subject_to(x[0] - x[1] <= 0, name="c3")
    m.subject_to(2 * x[0] == 1, name="half")
    m.minimize(0 * x[0])
    res = milp_iis(m)

    INF = highspy.HighsModelStatus.kInfeasible
    claims = [i for i, (pre, st) in enumerate(log) if pre != "off" and st == INF]
    assert claims, "no presolve-on infeasibility claim was exercised"
    for i in claims:
        assert i + 1 < len(log) and log[i + 1][0] == "off", (i, log)
    assert res.n_cross_checks == len(claims)
    assert res.n_solves == len(log)


@pytest.mark.smoke
@pytest.mark.parametrize("confirm", ["each", "final"])
@pytest.mark.parametrize("max_batch", [1, None])
def test_batch_and_single_filters_both_give_certified_iis(max_batch, confirm):
    """Adaptive batch deletion (and its one-at-a-time limit) stays sound and certified."""
    from discopt._iis_milp import milp_iis

    rng = np.random.default_rng(2093)
    n_models = n_checked = 0
    for _ in range(40):
        n, mrows = 5, 8
        A = rng.integers(-3, 4, size=(mrows, n)).astype(float)
        rhs = rng.integers(-4, 5, size=mrows).astype(float) + rng.choice([0.0, 0.5], mrows)
        lb, ub = np.zeros(n), np.full(n, 3.0)
        is_int = rng.random(n) < 0.6
        rlo, rhi = np.full(mrows, -np.inf), rhs.copy()
        if not _subsystem_infeasible(
            A, rlo, rhi, lb, ub, range(mrows), set(range(n)), set(range(n)), np.flatnonzero(is_int)
        ):
            continue
        m = dm.Model("r")
        xs = [
            (m.integer(f"x{j}", lb=0, ub=3) if is_int[j] else m.continuous(f"x{j}", lb=0, ub=3))
            for j in range(n)
        ]
        for r in range(mrows):
            m.subject_to(sum(A[r, j] * xs[j] for j in range(n)) <= rhs[r])
        m.minimize(0 * xs[0])
        res = milp_iis(m, time_limit=10, max_batch=max_batch, confirm=confirm)
        n_models += 1
        assert res.proven_irreducible
        kept_lb = {j for j, s in res.bounds if s == "lower"}
        kept_ub = {j for j, s in res.bounds if s == "upper"}
        args = (A, rlo, rhi, lb, ub, res.constraints, kept_lb, kept_ub, res.integrality)
        assert _subsystem_infeasible(*args)
        members = (
            [("constraint", r) for r in res.constraints]
            + [("bound", j, s) for j, s in res.bounds]
            + [("integrality", j) for j in res.integrality]
        )
        for key in members:
            x = res.certificates[key]
            ok = (
                A,
                rlo,
                rhi,
                lb,
                ub,
                is_int,
                res.constraints,
                kept_lb,
                kept_ub,
                res.integrality,
                x,
            )
            assert _independent_check(*ok, drop=key)
            assert not _independent_check(*ok)
            n_checked += 1
    print(
        f"[iis-batch max_batch={max_batch} confirm={confirm}] models={n_models} certs={n_checked}"
    )
    assert n_models >= 10 and n_checked >= 30, (n_models, n_checked)


@pytest.mark.smoke
def test_batching_cuts_solves_when_most_rows_are_redundant():
    from discopt._iis_milp import milp_iis

    def build():
        m = dm.Model("many")
        x = m.continuous("x", shape=(60,), lb=0, ub=10)
        m.subject_to(x[0] >= 5, name="big")
        for k in range(1, 59):
            m.subject_to(x[k] + x[k + 1] <= 15, name=f"rh{k}")
        m.subject_to(x[0] <= 2, name="small")
        m.minimize(0 * x[0])
        return m

    one = milp_iis(build(), max_batch=1)
    many = milp_iis(build())
    assert one.constraints == many.constraints == [0, 59]
    assert many.n_solves < one.n_solves / 3, (many.n_solves, one.n_solves)


@pytest.mark.smoke
def test_final_confirmation_catches_a_false_infeasible_verdict(monkeypatch):
    """confirm="final" acts on unconfirmed presolve-on verdicts, so a false one must be
    caught by the final two-configuration check and answered with a full "each" rerun."""
    from discopt import _iis_milp
    from discopt._iis_milp import milp_iis

    MS = highspy.HighsModelStatus
    real_solve = _iis_milp._Engine._solve
    lies = {"n": 0}

    def lying_solve(self, presolve):
        st = real_solve(self, presolve)
        # Once, in the final-mode engine only, report a FEASIBLE probe as infeasible.
        if not self.confirm_each and presolve == "choose" and st == MS.kOptimal and not lies["n"]:
            lies["n"] += 1
            return MS.kInfeasible
        return st

    monkeypatch.setattr(_iis_milp._Engine, "_solve", lying_solve)
    m = dm.Model("ex")
    x = m.integer("x", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] + x[1] <= 1.5, name="c1")
    m.subject_to(x[0] + x[1] >= 2.5, name="c2")
    m.subject_to(x[0] - x[1] <= 0, name="c3")
    m.subject_to(2 * x[0] == 1, name="half")
    m.minimize(0 * x[0])
    res = milp_iis(m, confirm="final")
    assert lies["n"] == 1, "the false verdict was never injected"
    assert res.fallback and res.confirm == "final"
    # The fallback result is the honest per-probe one: compare with a clean run.
    clean = milp_iis(m, confirm="each")
    assert (res.constraints, res.bounds, res.integrality) == (
        clean.constraints,
        clean.bounds,
        clean.integrality,
    )
    assert set(res.certificates) == set(clean.certificates)
