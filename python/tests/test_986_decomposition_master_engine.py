"""#986: a *linear* decomposition master must be solved exactly, and stay exact.

``nlp_solver`` selects the engine for the **NLP/LP recourse subproblem**. Three
sites reused it to route their **master** — a pure linear MILP — to the
POUNCE-IPM-backed B&B, whose returned points are interior rather than vertices
and whose objective is an analytic-centre value. That objective *is* the reported
dual ``bound``, so the conflation is a certificate defect, not a style one:
measured over 40 seeded two-stage instances on ``main``, 19 returned a ``bound``
strictly above the incumbent it certifies (worst +1.67e-07).

Pinning the masters to the exact simplex was tried once before (#977, PR #983
commit 53dbc17) and reverted, because on its own it makes classical Benders stall
out at ``iteration_limit`` with no incumbent at all. The mechanism, measured (the
#986 entry experiment) rather than assumed: an exact master returns a *vertex*,
whose sub-ulp violation of its own variable bounds (measured up to 1.8e-07, and
1.8e-15 on the instance in the reverted CI run) is scaled by the recourse row
coefficients into ``rhs = r - A_x x_hat``. A gated recourse row pair
(``y <= a*b``, ``y >= a*b``) then becomes *mutually inconsistent by ~1e-14*, an
exact LP presolve reports the recourse INFEASIBLE at a point where it is
feasible, and the block yields a feasibility cut that cannot separate instead of
an incumbent. The hypothesis recorded in #986 — degenerate recourse duals giving
a near-zero cut slope — was **falsified**: at every stalling iteration the
optimality cuts classify as converged (``eta_b >= Q_b``), i.e. the duals are fine.

So the tests below pin both halves: the masters are exact, *and* the classes that
stall when a master is exact still reach ``optimal``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from itertools import product

import numpy as np
import pytest

pytest.importorskip("pounce", reason="the decomposition solvers need an LP/MILP engine")

import discopt.modeling as dm  # noqa: E402
from discopt.solvers import SolveStatus, lp_backend  # noqa: E402

pytestmark = pytest.mark.requires_pounce


# ── corpus ────────────────────────────────────────────────────


def _two_stage(seed: int):
    """A two-stage MILP with capacity-gated recourse.

    Even seeds use the ``gate`` shape — a recourse variable pinned between
    ``lo*b`` and ``hi*b``, which collapses to a single point when the gating
    binary is 0. That is the structure whose recourse LP an exact presolve calls
    infeasible once the master point carries sub-ulp noise (#986). Odd seeds use
    the classic single-sided capacity form.
    """
    rng = np.random.default_rng(seed)
    gate = seed % 2 == 0
    nb = int(rng.integers(2, 6))
    nx = int(rng.integers(2, 6))
    m = dm.Model(f"ts{seed}")
    b = [m.binary(f"b{i}") for i in range(nb)]
    x = [m.continuous(f"x{i}", lb=0, ub=20) for i in range(nx)]
    if gate:
        for i in range(nx):
            j = i % nb
            m.subject_to(x[i] <= float(rng.uniform(6.0, 15.0)) * b[j])
            m.subject_to(x[i] >= float(rng.uniform(1.0, 5.0)) * b[j])
        m.subject_to(sum(b) >= max(1, nb // 2))
    else:
        cap = rng.uniform(3.0, 12.0, size=(nx, nb))
        dem = rng.uniform(1.0, 6.0, size=nx)
        for i in range(nx):
            m.subject_to(x[i] <= sum(float(cap[i, j]) * b[j] for j in range(nb)))
            m.subject_to(x[i] >= float(dem[i]))
        m.subject_to(sum(b) >= 1)
    cb = rng.uniform(1.0, 5.0, size=nb)
    cx = rng.uniform(0.5, 3.0, size=nx)
    m.minimize(
        sum(float(cx[i]) * x[i] for i in range(nx)) + sum(float(cb[j]) * b[j] for j in range(nb))
    )
    return m


def _enumerated_optimum(model):
    """The instance's true optimum by enumeration + exact LP, not by a solver.

    ``model.solve()`` is itself only accurate to ~1e-06 on these instances, so it
    cannot adjudicate a 1e-07 question about the Benders bound. Every first-stage
    assignment is enumerated and its recourse solved on the exact simplex.
    """
    from discopt.decomposition._linear import extract_linear
    from discopt.decomposition.benders.solver import _partition_columns
    from discopt.decomposition.structure import detect_decomposition, flat_bounds

    spx = lp_backend.get_exact_lp_solver()
    if spx is None:
        pytest.skip("no exact simplex oracle available")
    lin = extract_linear(model)
    part = _partition_columns(model, detect_decomposition(model))
    mcols, scols = part.master_cols, part.sub_cols
    lb_all, ub_all = flat_bounds(model)
    A_leq, b_leq, _ = lin.rows_leq()
    A = np.asarray(A_leq, dtype=float)
    b = np.asarray(b_leq, dtype=float)
    A_sub = A[:, scols]
    recourse_rows = np.any(np.abs(A_sub) > 0, axis=1)
    sub_bounds = [(float(lb_all[j]), float(ub_all[j])) for j in scols]
    best = None
    for assign in product([0.0, 1.0], repeat=len(mcols)):
        x_m = np.array(assign, dtype=float)
        rhs = b - A[:, mcols] @ x_m
        if np.any(rhs[~recourse_rows] < -1e-9):
            continue  # master-only row violated
        r = spx(lin.c[scols], A_ub=A_sub[recourse_rows], b_ub=rhs[recourse_rows], bounds=sub_bounds)
        if r.status != SolveStatus.OPTIMAL:
            continue
        val = float(lin.c[mcols] @ x_m) + float(r.objective) + lin.c_offset
        if best is None or val < best:
            best = val
    return best


# ── the three sites are routed to the exact engine ────────────


def _record_milp_selection(monkeypatch):
    """Record every ``get_milp_solver`` selection the solvers make."""
    selections: list[dict] = []
    original = lp_backend.get_milp_solver

    def spy(prefer_pounce: bool = False, backend: str = "auto"):
        selections.append({"prefer_pounce": prefer_pounce, "backend": backend})
        return original(prefer_pounce=prefer_pounce, backend=backend)

    monkeypatch.setattr(lp_backend, "get_milp_solver", spy)
    return selections


def test_benders_master_is_not_routed_by_nlp_solver(monkeypatch):
    selections = _record_milp_selection(monkeypatch)
    from discopt.decomposition.benders.solver import solve_benders

    solve_benders(_two_stage(0), nlp_solver="pounce", max_iterations=50)

    assert selections, "no MILP engine was selected; the test measured nothing"
    for sel in selections:
        assert sel["backend"] == "simplex", (
            f"the Benders master asked for backend={sel['backend']!r}; a linear "
            "master must be pinned to the exact-vertex engine (#986)"
        )
        assert not sel["prefer_pounce"], "the master is still routed by nlp_solver (#986)"


def test_lagrangian_master_is_not_routed_by_nlp_solver(monkeypatch):
    selections = _record_milp_selection(monkeypatch)
    from discopt.decomposition.lagrangian.solver import solve_lagrangian

    m = _two_stage(1)
    solve_lagrangian(m, nlp_solver="pounce", max_iterations=3, time_limit=30.0)

    assert selections, "no MILP engine was selected; the test measured nothing"
    for sel in selections:
        assert sel["backend"] == "simplex", (
            f"the Lagrangian subproblem asked for backend={sel['backend']!r}; its "
            "bound is the reported certificate and must come from the exact engine"
        )
        assert not sel["prefer_pounce"]


def test_lagrangian_node_bounder_is_not_routed_by_nlp_solver(monkeypatch):
    selections = _record_milp_selection(monkeypatch)
    from discopt.decomposition.lagrangian.node_bounder import LagrangianNodeBounder

    m = _two_stage(3)
    bounder = LagrangianNodeBounder.try_build(m)
    if bounder is None:
        pytest.skip("the node-bounder hook does not apply to this instance")

    assert selections, "no MILP engine was selected; the test measured nothing"
    for sel in selections:
        assert sel["backend"] == "simplex", (
            f"the node bounder asked for backend={sel['backend']!r}; its objective "
            "is a node lower bound and must come from the exact engine"
        )
        assert not sel["prefer_pounce"]


# ── and the certificate the exact master now reports is sound ──


def test_benders_bound_never_exceeds_the_incumbent_it_certifies():
    """``bound <= incumbent`` on the classical-Benders path.

    Live on ``main``: 19 of these 40 instances returned ``bound > objective``,
    worst +1.67e-07. The master objective is the reported bound, so an
    analytic-centre master value drifts straight into the certificate; the cut
    anchors have the same problem one level down when the recourse duals come from
    an interior-point solve that satisfies ``c - A^T lam - rc = 0`` only to its
    convergence tolerance.
    """
    from discopt.decomposition.benders.solver import solve_benders

    checked = 0
    violations = []
    for seed in range(40):
        res = solve_benders(_two_stage(seed), max_iterations=200)
        if res.bound is None or res.objective is None:
            continue
        checked += 1
        scale = max(1.0, abs(float(res.objective)))
        excess = float(res.bound) - float(res.objective)
        if excess > 1e-9 * scale:
            violations.append((seed, float(res.bound), float(res.objective), excess))

    assert checked >= 30, f"only {checked} certificate comparisons executed; expected >= 30"
    assert not violations, f"bound > incumbent on {len(violations)} instances: {violations[:5]}"


def test_benders_gate_structured_recourse_still_reaches_optimal():
    """The class that stalls when the master is exact but its noise is unguarded.

    This is the regression the reverted PR #983 commit tripped: with an exact
    master and no clamp on the master point, a gated recourse row pair becomes
    inconsistent by ~1e-14, the recourse is reported INFEASIBLE at a feasible
    point, and the run ends at ``iteration_limit`` with ``objective is None``.
    """
    from discopt.decomposition.benders.solver import solve_benders

    checked = 0
    for seed in range(0, 40, 2):  # even seeds are the gated shape
        model = _two_stage(seed)
        truth = _enumerated_optimum(model)
        if truth is None:
            continue
        res = solve_benders(model, max_iterations=200)
        checked += 1
        assert res.objective is not None, (
            f"seed {seed}: no incumbent at all (status={res.status}) — the recourse "
            "was declared infeasible at a feasible master point (#986)"
        )
        assert res.status == "optimal", f"seed {seed}: status={res.status}, expected optimal"
        assert float(res.objective) == pytest.approx(truth, rel=1e-6, abs=1e-6)
        assert res.bound is not None and float(res.bound) <= truth + 1e-6 * max(1.0, abs(truth))

    assert checked >= 15, f"only {checked} gated instances exercised; expected >= 15"
