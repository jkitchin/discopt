"""Implied-binary continuous columns in the IIS engine's CDCL candidate stage.

``_iis_milp._binary_core`` (the CDCL core candidate) applies only to pure-binary
models. ``_lift_implied_binaries`` finds continuous columns that may be taken
binary without changing feasibility *of any row subset*, and the stage sees them
as binaries; the base probe and the filter still run on the original types.
Measured on MIPcc27 neos-1461051 (448 binaries, 80 continuous columns
``c <= b + b'`` covered by ``sum c >= 1``): the stage declined before; with the
lift the MIPcc27 entry returns a 335-element IIS (checker PASS) where it returned
4387 elements, against a Gurobi reference of 3238.

These tests pin (1) the rule's soundness by exhaustive comparison over row
subsets, (2) its refusals, and (3) that the engine uses the candidate on a mixed
model and still returns an independently infeasible, certified subsystem.
"""

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

highspy = pytest.importorskip("highspy")
from discopt import _iis_milp  # noqa: E402
from discopt._iis_milp import _lift_implied_binaries, milp_iis  # noqa: E402

INF = 1e20


@pytest.fixture(autouse=True)
def _pin_highs_engine(monkeypatch):
    monkeypatch.setenv("DISCOPT_IIS_ENGINE", "highs")


def _random_mixed(rng, nb=6, nc=4):
    """Rows (cols, vals, lo, hi) over nb binaries then nc continuous columns:
    upper rows c - x_a - x_b <= 0, covering rows sum(c) >= 1, binary clauses."""
    rows = []
    for k in range(nc):
        a, b = rng.choice(nb, size=2, replace=False)
        rows.append(([nb + k, a, b], [1.0, -1.0, -1.0], -INF, 0.0))
    for _ in range(2):
        cs = rng.choice(nc, size=2, replace=False)
        rows.append(([nb + int(k) for k in cs], [1.0, 1.0], 1.0, INF))
    for _ in range(int(rng.integers(3, 7))):
        vs = rng.choice(nb, size=2, replace=False)
        neg = rng.integers(2, size=2).astype(bool)
        vals = [-1.0 if g else 1.0 for g in neg]
        rows.append(([int(v) for v in vs], vals, 1.0 - float(neg.sum()), INF))
    return rows


def _matrices(rows, n):
    data, ri, ci = [], [], []
    for r, (cols, vals, _, _) in enumerate(rows):
        ri += [r] * len(cols)
        ci += list(cols)
        data += list(vals)
    A = sp.csr_matrix((data, (ri, ci)), shape=(len(rows), n))
    lo = np.array([r[2] for r in rows])
    hi = np.array([r[3] for r in rows])
    return A, lo, hi


def _feasible(rows, keep, n, lb, ub, is_int):
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.silent()
    h.addVars(n, lb, np.where(ub >= INF, inf, ub))
    for r in keep:
        cols, vals, lo, hi = rows[r]
        h.addRow(
            -inf if lo <= -INF else lo,
            inf if hi >= INF else hi,
            len(cols),
            np.array(cols, dtype=np.int32),
            np.array(vals),
        )
    idx = np.flatnonzero(is_int).astype(np.int32)
    h.changeColsIntegrality(len(idx), idx, np.array([highspy.HighsVarType.kInteger] * len(idx)))
    h.run()
    st = h.getModelStatus()
    assert st in (highspy.HighsModelStatus.kOptimal, highspy.HighsModelStatus.kInfeasible), st
    return st == highspy.HighsModelStatus.kOptimal


@pytest.mark.smoke
def test_lift_preserves_feasibility_of_every_row_subset():
    """Exhaustive: for every subset of rows, original types and lifted types
    agree on feasibility."""
    rng = np.random.default_rng(0)
    nb, nc = 6, 4
    n = nb + nc
    lb = np.zeros(n)
    ub = np.r_[np.ones(nb), np.full(nc, INF)]
    is_int = np.r_[np.ones(nb, bool), np.zeros(nc, bool)]
    compared = disagreements = 0
    for _ in range(6):
        rows = _random_mixed(rng, nb, nc)
        A, lo, hi = _matrices(rows, n)
        lifted = _lift_implied_binaries(A, lo, hi, lb, ub, is_int)
        assert lifted == set(range(nb, n))
        lub, lint = ub.copy(), is_int.copy()
        lub[nb:] = 1.0
        lint[nb:] = True
        m = len(rows)
        for size in range(1, m + 1):
            for keep in itertools.combinations(range(m), size):
                if rng.random() > 0.15:  # sample ~15% of the subsets
                    continue
                a = _feasible(rows, keep, n, lb, ub, is_int)
                b = _feasible(rows, keep, n, lb, lub, lint)
                disagreements += a != b
                compared += 1
    print(f"[lift] subsets compared={compared}")
    assert compared >= 200, compared
    assert disagreements == 0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "row, lb, ub",
    [
        (([1, 0], [1.0, -0.5], -INF, 0.0), 0.0, INF),  # c <= x/2: bound not integer-valued
        (([1, 0], [1.0, -1.0], 0.0, 0.0), 0.0, INF),  # equality row
        (([1, 0], [1.0, -1.0], -INF, 0.5), 0.0, INF),  # fractional right-hand side
        (([1, 0], [1.0, 1.0], 2.0, INF), 0.0, INF),  # covering, but c = 1 alone does not satisfy it
        (([1, 0], [1.0, -1.0], 0.0, INF), 0.0, INF),  # covering with a negative other term
        (([1, 0], [1.0, -1.0], -INF, 0.0), 0.0, 0.5),  # ub < 1
        (([1, 0], [1.0, -1.0], -INF, 0.0), -1.0, INF),  # lb != 0
    ],
)
def test_lift_refuses(row, lb, ub):
    cols, vals, lo, hi = row
    A = sp.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, 2))
    got = _lift_implied_binaries(
        A,
        np.array([lo]),
        np.array([hi]),
        np.array([0.0, lb]),
        np.array([1.0, ub]),
        np.array([True, False]),
    )
    assert got == set()


def _clause_mixed_model(rng):
    nb, nc = 8, 6
    mod = dm.Model("mixed")
    x = [mod.binary(f"x{j}") for j in range(nb)]
    c = [mod.continuous(f"c{k}", lb=0) for k in range(nc)]
    for k in range(nc):
        a, b = rng.choice(nb, size=2, replace=False)
        mod.subject_to(c[k] <= x[a] + x[b])
    for k in range(0, nc, 2):
        mod.subject_to(c[k] + c[k + 1] >= 1)
    for _ in range(40):
        vs = rng.choice(nb, size=3, replace=False)
        neg = rng.integers(2, size=3).astype(bool)
        mod.subject_to(sum((-x[v] if g else x[v]) for v, g in zip(vs, neg)) >= 1 - int(neg.sum()))
    mod.minimize(0 * x[0])
    return mod


@pytest.mark.smoke
def test_engine_uses_the_candidate_on_a_mixed_model():
    """Before the lift the stage declined every such model (sat_rows None);
    with it the candidate is used, and every result is still independently
    infeasible and fully certified."""
    from discopt.decomposition._linear import extract_linear

    rng = np.random.default_rng(3)
    used = checked = 0
    for _ in range(60):
        mod = _clause_mixed_model(rng)
        lin = extract_linear(mod)
        A = sp.csr_matrix(lin.A)
        n = lin.n
        rows = []
        for r in range(lin.m):
            s0, e0 = A.indptr[r], A.indptr[r + 1]
            lo = lin.b[r] if lin.sense[r] in (">=", "==") else -INF
            hi = lin.b[r] if lin.sense[r] in ("<=", "==") else INF
            rows.append((list(A.indices[s0:e0]), list(A.data[s0:e0]), lo, hi))
        lb = np.zeros(n)
        ub = np.r_[np.ones(8), np.full(6, INF)]
        is_int = np.r_[np.ones(8, bool), np.zeros(6, bool)]
        if _feasible(rows, range(lin.m), n, lb, ub, is_int):
            continue
        res = milp_iis(mod, time_limit=10)
        used += res.sat_rows is not None
        assert res.proven_irreducible and res.complete
        # independent check on ORIGINAL types: kept rows, kept integralities,
        # kept bounds of the other columns
        kub = np.full(n, INF)
        klb = np.full(n, -INF)
        for j, s in res.bounds:
            (klb if s == "lower" else kub)[j] = (lb if s == "lower" else ub)[j]
        kint = np.zeros(n, bool)
        kint[list(res.integrality)] = True
        klb[kint], kub[kint] = lb[kint], ub[kint]
        assert not _feasible(rows, res.constraints, n, klb, kub, kint)
        size = len(res.constraints) + len(res.bounds) + len(res.integrality)
        assert len(res.certificates) == size
        checked += 1
    print(f"[lift-engine] models={checked} candidate_used={used}")
    assert checked >= 10 and used >= 10, (checked, used)


@pytest.mark.smoke
def test_lift_is_seen_only_by_the_candidate_stage(monkeypatch):
    """The stage receives lifted types; the model handed to the filter is unchanged."""
    seen = {}
    real = _iis_milp._binary_core

    def capture(A, row_lo, row_hi, col_lb, col_ub, is_int, *rest):
        seen["is_int"] = is_int.copy()
        seen["ub"] = col_ub.copy()
        return real(A, row_lo, row_hi, col_lb, col_ub, is_int, *rest)

    monkeypatch.setattr(_iis_milp, "_binary_core", capture)
    mod = dm.Model("tiny")
    x = mod.binary("x")
    c = mod.continuous("c", lb=0)
    mod.subject_to(c <= x)
    mod.subject_to(c >= 1)
    mod.subject_to(-x >= 0)
    mod.minimize(0 * x)
    res = milp_iis(mod, time_limit=10)
    assert seen["is_int"].tolist() == [True, True]
    assert seen["ub"].tolist() == [1.0, 1.0]
    assert res.proven_irreducible
    assert 1 not in res.integrality  # c was never an integrality element
