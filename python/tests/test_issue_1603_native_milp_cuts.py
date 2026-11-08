"""#1603: ``milp_backend="native"`` certified a wrong ``optimal`` through an invalid root cut.

Two derivation defects in the native MILP route's root cut loop, one class (a cut
derived from a row or edge whose terms were not all accounted for):

1. **Clique edges in the wrong index space.** The Rust presolve clique pass
   (``crates/discopt-core/src/presolve/cliques.rs``) reported each conflict edge
   as a pair of variable-BLOCK indices, while ``_root_cover_cut_loop`` reads them
   as FLAT columns. Any array variable declared before the binaries shifts the two
   spaces apart, so the edge for ``s0 + s1 == 1`` (blocks 4, 5) became the clique
   cut ``d[0] + d[1] <= 1`` on two continuous columns, cutting the optimum
   (``d = ub``) off: ``optimal`` 1.895 against the true 4.275.
2. **Cover rows with silently dropped tiny coefficients.** The Python cover
   separator dropped nonzeros at or below its tolerance before testing whether a
   row is a binary knapsack, so ``b0 + b1 - 1e-7 y <= 1.5`` (``y in [0, 1e9]``)
   was treated as ``b0 + b1 <= 1.5`` and gave the invalid cover ``b0 + b1 <= 1``.
   The same separator runs in the convex MINLP NLP-BB root-cut stage.

Plus the backstop: a cut that a verified-feasible point violates is rejected and
counted, never installed.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._relax.cover_cuts import has_binary_knapsack_rows, separate_cover_cuts

pytest.importorskip("highspy")

_ROUTES = (
    {},  # default: pure MILP -> HiGHS
    {"milp_backend": "native"},
    {"milp_backend": "native", "milp_cuts": False},
)


def _issue_repro():
    """The #1603 repro: an array variable ``d`` declared before scalar binaries."""
    m = dm.Model("issue1603")
    x = m.binary("x", shape=(4,))
    d = m.continuous("d", shape=(4,), lb=0, ub=[75, 600, 175, 75])
    h = m.binary("h")
    m.binary("z")
    s0 = m.binary("s0")
    s1 = m.binary("s1")
    for i, (a, b) in enumerate([(45.45, 0.45), (404, 4), (136.35, 1.35), (50.5, 0.5)]):
        m.subject_to(a * x[i] - d[i] <= b)
    m.subject_to(s0 + s1 == 1)
    m.maximize(0.02 * d[0] + 0.0015 * d[1] + 0.006 * d[2] + 0.011 * d[3] - 0.05 * h)
    return m


def _tiny_coefficient_milp():
    """A binary knapsack row carrying a tiny continuous term with a huge range."""
    m = dm.Model("tiny_coef")
    b0 = m.binary("b0")
    b1 = m.binary("b1")
    y = m.continuous("y", lb=0.0, ub=1e9)
    b2 = m.binary("b2")
    b3 = m.binary("b3")
    m.subject_to(b0 + b1 - 1e-7 * y <= 1.5)
    m.subject_to(2 * b2 + 2 * b3 <= 3)
    m.maximize(b0 + b1 + b2 + b3 - 1e-12 * y)
    return m


@pytest.mark.parametrize("kw", _ROUTES, ids=["highs", "native", "native-nocuts"])
def test_issue_repro_all_routes(kw):
    r = _issue_repro().solve(time_limit=60, **kw)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(4.275, abs=1e-6)


@pytest.mark.parametrize("kw", _ROUTES, ids=["highs", "native", "native-nocuts"])
def test_tiny_coefficient_cover_all_routes(kw):
    # y = 5e6 lets b0 = b1 = 1, and the second row admits one of b2/b3: the optimum
    # is 3 - 5e-6. (HiGHS may report it as "feasible" with an NS-safe bound, #1509;
    # the value is what is checked.) Before the fix, native+cuts returned 2.0.
    r = _tiny_coefficient_milp().solve(time_limit=60, **kw)
    assert r.status in ("optimal", "feasible"), r.status
    assert r.objective == pytest.approx(2.999995, abs=1e-5)


def test_clique_edges_are_flat_binary_columns():
    m = _issue_repro()
    edges = S._extract_clique_edges(m)
    # x: 0..3, d: 4..7, h: 8, z: 9, s0: 10, s1: 11.
    assert (10, 11) in [tuple(sorted(e)) for e in edges], edges
    is_bin = S._binary_mask(m, 12)
    checked = 0
    for i, j in edges:
        assert is_bin[i] and is_bin[j], (i, j)
        checked += 1
    assert checked > 0


def test_non_binary_clique_edge_is_refused_loudly(monkeypatch):
    # Python-MILP-tree internals: ``milp_backend="native"`` is the Rust engine since the
    # retirement's R1 step, so the tree is reached through its opt-out until
    # ``_solve_milp_bb`` is deleted (docs/dev/python-milp-tree-retirement.md).
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", "0")
    # The old block-index edge (4, 5) names continuous columns d[0], d[1].
    monkeypatch.setattr(S, "_extract_clique_edges", lambda model: [(4, 5)])
    with pytest.raises(RuntimeError, match="binary flat columns"):
        _issue_repro().solve(time_limit=30, milp_backend="native")


def test_backstop_rejects_cut_violated_by_verified_point(monkeypatch):
    """A separator that emits the old invalid cut is caught by the incumbent screen."""
    # Python-MILP-tree internals: ``milp_backend="native"`` is the Rust engine since the
    # retirement's R1 step, so the tree is reached through its opt-out until
    # ``_solve_milp_bb`` is deleted (docs/dev/python-milp-tree-retirement.md).
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", "0")
    import discopt._relax.cover_cuts as CC

    fired = [0]

    def bogus(edges, x_star, tol=1e-6):
        fired[0] += 1
        return [(frozenset({4, 5}), 1.0)]  # d[0] + d[1] <= 1: the #1603 cut

    monkeypatch.setattr(CC, "separate_clique_cuts", bogus)
    # The known optimum as a warm start; the screen verifies it before using it.
    m = _issue_repro()
    v = m._variables  # x, d, h, z, s0, s1
    start = {v[0]: [0, 0, 0, 0], v[1]: [75, 600, 175, 75], v[2]: 0, v[3]: 0, v[4]: 1, v[5]: 0}
    r = m.solve(time_limit=60, milp_backend="native", initial_solution=start)
    assert fired[0] > 0, "the bogus separator never ran -- the probe did not fire"
    assert r.objective == pytest.approx(4.275, abs=1e-6)
    assert (r.solver_stats or {}).get("cuts/rejected_by_incumbent", 0) > 0, r.solver_stats


def test_verified_screen_points_refuse_infeasible_candidates():
    m = _issue_repro()
    n_vars, _lb, _ub, offs, sizes = S._extract_variable_info(m)
    good = np.array([0, 0, 0, 0, 75, 600, 175, 75, 0, 0, 1, 0], dtype=float)
    out_of_box = good.copy()
    out_of_box[4] = 80.0
    row_infeasible = good.copy()
    row_infeasible[10] = 0.0  # s0 + s1 == 1 violated
    fractional = good.copy()
    fractional[0] = 0.5
    pts = S._verified_cut_screen_points(
        m, [None, good, out_of_box, row_infeasible, fractional], n_vars, 12, offs, sizes
    )
    assert len(pts) == 1
    np.testing.assert_array_equal(pts[0], good)


# ---------------------------------------------------------------------------
# Cover separator: tiny nonzeros are charged to the capacity, not dropped
# ---------------------------------------------------------------------------
class TestCoverTinyCoefficients:
    A = np.array([[1.0, 1.0, -1e-7]])
    b = np.array([1.5])
    is_bin = np.array([True, True, False])
    x = np.array([1.0, 1.0, 0.0])

    def test_unbounded_tiny_term_refuses_the_row(self):
        assert separate_cover_cuts(self.A, self.b, self.x, self.is_bin) == []
        assert not has_binary_knapsack_rows(self.A, self.b, self.is_bin)

    def test_large_range_tiny_term_refuses_cover(self):
        lb, ub = np.array([0, 0, 0.0]), np.array([1, 1, 1e9])
        # cap = 1.5 + 1e-7 * 1e9 = 101.5: no cover, so no cut.
        assert separate_cover_cuts(self.A, self.b, self.x, self.is_bin, lb=lb, ub=ub) == []

    def test_small_range_tiny_term_still_covers(self):
        lb, ub = np.array([0, 0, 0.0]), np.array([1, 1, 1.0])
        # cap = 1.5 + 1e-7: {0, 1} is still a cover, and it is valid.
        cuts = separate_cover_cuts(self.A, self.b, self.x, self.is_bin, lb=lb, ub=ub)
        assert cuts == [(frozenset({0, 1}), 1.0)]

    def test_cover_never_cuts_a_feasible_point_with_tiny_terms(self):
        rng = np.random.default_rng(1603)
        checked = 0
        for _ in range(300):
            nb = int(rng.integers(2, 5))
            a = np.concatenate([rng.integers(1, 5, nb).astype(float), [rng.choice([-1, 1]) * 1e-7]])
            yub = float(rng.choice([1.0, 1e3, 1e9]))
            b = float(rng.uniform(a[:nb].min(), a[:nb].sum()))
            lb = np.zeros(nb + 1)
            ub = np.concatenate([np.ones(nb), [yub]])
            is_bin = np.concatenate([np.ones(nb, bool), [False]])
            cuts = separate_cover_cuts(
                a.reshape(1, -1), np.array([b]), rng.random(nb + 1), is_bin, lb=lb, ub=ub
            )
            for bits in np.ndindex(*(2,) * nb):
                xb = np.array(bits, float)
                # most favourable y for feasibility of the row
                y = 0.0 if a[-1] > 0 else yub
                if a[:nb] @ xb + a[-1] * y > b + 1e-12:
                    continue
                for cover, rhs in cuts:
                    checked += 1
                    assert sum(xb[j] for j in cover) <= rhs + 1e-9, (a, b, yub, bits, cover)
        print("cover feasible-point checks:", checked)
        assert checked > 0


# ---------------------------------------------------------------------------
# MINLP: the convex NLP-BB root-cut stage shares the cover separator
# ---------------------------------------------------------------------------
def test_nlpbb_root_cut_stage_bound_is_sound_with_tiny_term(monkeypatch):
    from discopt._relax.nlp_evaluator import NLPEvaluator
    from discopt.solvers._root_cuts import generate_root_cuts

    monkeypatch.setenv("DISCOPT_NLPBB_ROOT_CUTS", "1")

    def build():
        m = dm.Model("minlp_tiny")
        b0 = m.binary("b0")
        b1 = m.binary("b1")
        y = m.continuous("y", lb=0, ub=1e9)
        t = m.continuous("t", lb=0, ub=1e6)
        b2 = m.binary("b2")
        b3 = m.binary("b3")
        m.subject_to(b0 + b1 - 1e-7 * y <= 1.5)
        m.subject_to(2 * b2 + 2 * b3 <= 3)
        m.subject_to(t >= (1e-7 * y) ** 2)
        m.minimize(-(b0 + b1 + b2 + b3) + 0.1 * t)
        return m

    # Optimum: b0 = b1 = 1, y = 5e6 (t = 0.25), one of b2/b3 -> -3 + 0.025 = -2.975.
    opt = -2.975
    x_opt = np.array([1.0, 1.0, 5e6, 0.25, 1.0, 0.0])
    m = build()
    lb = np.array([0, 0, 0, 0, 0, 0.0])
    ub = np.array([1, 1, 1e9, 1e6, 1, 1.0])
    is_int = np.array([1, 1, 0, 0, 1, 1], bool)
    res = generate_root_cuts(m, NLPEvaluator(m), lb, ub, is_int, is_int.copy())
    assert res.lp_bound is not None
    assert res.lp_bound <= opt + 1e-6, f"root LP bound {res.lp_bound} above the optimum {opt}"
    checked = 0
    for alpha, rhs in res.cuts:
        checked += 1
        assert float(np.asarray(alpha) @ x_opt) <= rhs + 1e-6 * (1 + abs(rhs)), (alpha, rhs)
    print("cuts checked at the optimum:", checked)


# ---------------------------------------------------------------------------
# Randomized differential panel: native+cuts vs native-nocuts vs HiGHS
# ---------------------------------------------------------------------------
def _random_milp(seed: int):
    """Small mixed binary/continuous MILP, feasible by construction.

    Exercises the #1603 class: array continuous variables declared before the
    binaries (index-space offsets), set-packing / partitioning rows (clique
    edges), mixed ``<=``/``>=`` rows, occasional tiny coefficients on
    wide-range continuous columns (cover accounting), and min/max senses.
    """
    rng = np.random.default_rng(seed)
    m = dm.Model(f"rand{seed}")
    nc = int(rng.integers(1, 4))
    nb = int(rng.integers(3, 7))
    cub = rng.choice([10.0, 100.0, 1e6], size=nc)
    c_first = bool(rng.integers(0, 2))
    if c_first:
        c = m.continuous("c", shape=(nc,), lb=0, ub=cub)
        b = m.binary("b", shape=(nb,))
    else:
        b = m.binary("b", shape=(nb,))
        c = m.continuous("c", shape=(nc,), lb=0, ub=cub)
    extra = [m.binary(f"s{k}") for k in range(int(rng.integers(0, 3)))]
    bins = [b[i] for i in range(nb)] + extra
    # A feasible anchor point.
    xb = rng.integers(0, 2, len(bins)).astype(float)
    xc = rng.uniform(0, 1, nc) * np.minimum(cub, 50.0)
    if extra:
        xb[nb:] = 0.0
        xb[nb] = 1.0
        m.subject_to(sum(extra) == 1)  # partitioning row -> clique edges
    for _ in range(int(rng.integers(2, 6))):
        kind = rng.integers(0, 4)
        if kind == 0:
            # set packing over a pair of binaries the anchor allows
            i, j = rng.choice(len(bins), 2, replace=False)
            if xb[i] + xb[j] <= 1:
                m.subject_to(bins[i] + bins[j] <= 1)
            continue
        ab = rng.integers(-5, 6, len(bins)).astype(float)
        ac = rng.integers(-5, 6, nc).astype(float)
        if kind == 1:
            k = int(rng.integers(0, nc))
            ac[k] = float(rng.choice([-1e-7, 1e-7]))
        expr = sum(float(ab[i]) * bins[i] for i in range(len(bins)) if ab[i] != 0)
        expr = expr + sum(float(ac[k]) * c[k] for k in range(nc) if ac[k] != 0)
        if isinstance(expr, (int, float)):
            continue
        act = float(ab @ xb + ac @ xc)
        slack = float(rng.uniform(0, 3))
        if rng.integers(0, 2):
            m.subject_to(expr <= act + slack)
        else:
            m.subject_to(expr >= act - slack)
    ob = rng.normal(size=len(bins))
    oc = rng.normal(size=nc) * rng.choice([1.0, 0.01], size=nc)
    obj = sum(float(ob[i]) * bins[i] for i in range(len(bins))) + sum(
        float(oc[k]) * c[k] for k in range(nc)
    )
    if rng.integers(0, 2):
        m.maximize(obj)
    else:
        m.minimize(obj)
    return m


def _close(a, b):
    return abs(a - b) <= 1e-6 + 1e-4 * max(abs(a), abs(b))


@pytest.mark.slow
def test_randomized_native_cut_panel():
    """200 seeded MILPs: native+cuts and native-nocuts against HiGHS.

    Measured at f5c4c57e (main, before the fix): native+cuts got 6/200 wrong
    (4 wrong ``optimal`` objectives, 2 false ``infeasible``); native-nocuts 0/200.
    After the fix: 0/200 wrong on both.

    A loud refusal from the #952 exit gate is recorded separately from a wrong
    answer, and since #1606 there must be none. Before it, two instances (seeds 73,
    195) ended there on native+cuts -- an integral incumbent ~1.2e-6 off an
    original row on a tight but VALID GMI face. The exit now re-derives such an
    incumbent's continuous columns on the declared rows (see
    ``test_issue_1606_lp_residual_incumbent_is_repolished``).
    """
    n_seeds = 200
    compared = 0
    wrong = []
    refused = []
    for seed in range(n_seeds):
        res = {}
        for name, kw in zip(("highs", "native", "nocuts"), _ROUTES):
            try:
                r = _random_milp(seed).solve(time_limit=30, **kw)
                res[name] = (r.status, r.objective)
            except RuntimeError as e:
                if "MILP-BB returned an infeasible point labeled" not in str(e):
                    raise
                res[name] = ("refused", None)
        ref = res["highs"]
        assert ref[0] in ("optimal", "feasible"), (seed, ref)
        for name in ("native", "nocuts"):
            st, obj = res[name]
            compared += 1
            if st == "refused":
                refused.append((seed, name))
            elif st != "optimal" or not _close(obj, ref[1]):
                wrong.append((seed, name, st, obj, ref[1]))
    print(
        f"panel: {n_seeds} instances, {compared} comparisons, {len(wrong)} wrong, "
        f"{len(refused)} exit-gate refusals {refused}"
    )
    for w in wrong:
        print("  WRONG", w)
    assert compared > 0
    assert not wrong, wrong
    assert not refused, refused  # #1606: zero exit-gate refusals


# ---------------------------------------------------------------------------
# #1607 review: the cut screen judges the SNAPPED point
# ---------------------------------------------------------------------------
def _two_packing_rows():
    m = dm.Model("screen_snap")
    b = [m.binary(f"b{i}") for i in range(3)]
    m.subject_to(2 * b[0] + 2 * b[1] <= 3)
    m.subject_to(2 * b[1] + 2 * b[2] <= 3)
    m.maximize(b[0] + b[1] + b[2])
    return m, b


def test_verified_screen_points_snap_integer_columns():
    m, _ = _two_packing_rows()
    n_vars, _lb, _ub, offs, sizes = S._extract_variable_info(m)
    near = np.array([1.0, 9e-6, 1.0 - 4e-6])
    pts = S._verified_cut_screen_points(m, [near], n_vars, 3, offs, sizes)
    assert len(pts) == 1
    np.testing.assert_array_equal(pts[0], [1.0, 0.0, 1.0])
    assert near[1] == 9e-6, "the caller's candidate must not be modified in place"


def test_near_integral_warm_start_does_not_reject_valid_cuts(caplog):
    """``b1 = 9e-6`` passes the 1e-5 integrality test. Screened unsnapped, it
    rejected 5 valid root cuts and logged a false "separator defect" WARNING."""
    m, b = _two_packing_rows()
    start = {b[0]: 1.0, b[1]: 9e-6, b[2]: 1.0}
    with caplog.at_level("WARNING", logger="discopt.solver"):
        r = m.solve(time_limit=30, milp_backend="native", initial_solution=start)
    stats = r.solver_stats or {}
    assert r.status == "optimal" and r.objective == pytest.approx(2.0, abs=1e-9)
    installed = sum(v for k, v in stats.items() if k.startswith("cuts/") and "rejected" not in k)
    assert installed > 0, f"no root cut was installed, so nothing was screened: {stats}"
    assert stats.get("cuts/rejected_by_incumbent", 0) == 0, stats
    assert not [rec for rec in caplog.records if "separator defect" in rec.getMessage()]


# ---------------------------------------------------------------------------
# #1606: an integral incumbent a node-LP residual off a declared row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [73, 195])
def test_issue_1606_lp_residual_incumbent_is_repolished(seed, monkeypatch):
    """Native+cuts used to raise the #952 refusal here: the tree's integral
    incumbent sat on a tight valid GMI face, 1.15e-6 (seed 73) / 1.29e-6 (seed 195)
    off an original row. The exit now re-solves its continuous columns with the
    integers fixed, over the declared rows, and the gate accepts the result."""
    # Python-MILP-tree internals: ``milp_backend="native"`` is the Rust engine since the
    # retirement's R1 step, so the tree is reached through its opt-out until
    # ``_solve_milp_bb`` is deleted (docs/dev/python-milp-tree-retirement.md).
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", "0")
    ref = _random_milp(seed).solve(time_limit=30)
    r = _random_milp(seed).solve(time_limit=30, milp_backend="native")
    assert r.algorithm_route.startswith("native-milp: discopt MILP branch and bound"), (
        r.algorithm_route
    )
    assert (r.solver_stats or {}).get("incumbent/repolished") == 1.0, (
        "the repolish arm did not fire -- this no longer exercises #1606",
        r.solver_stats,
    )
    assert r.status == "optimal", r.status
    assert _close(r.objective, ref.objective), (r.objective, ref.objective)
    # The certificate: the bound never crosses the reported objective.
    sense = 1.0 if seed == 73 else -1.0  # 73 minimises, 195 maximises
    assert sense * (r.bound - r.objective) <= 1e-9, (r.bound, r.objective)


@pytest.mark.parametrize("seed", [73, 195])
def test_issue_1606_instances_certify_on_the_rust_engine(seed):
    """The user-visible half of #1606 on today's ``milp_backend="native"`` route (the
    Rust engine): the same instances certify the reference optimum, bound on the
    right side of the objective."""
    ref = _random_milp(seed).solve(time_limit=30)
    r = _random_milp(seed).solve(time_limit=30, milp_backend="native")
    assert r.algorithm_route.startswith("native-milp: discopt monolithic Rust MILP"), (
        r.algorithm_route
    )
    assert r.status == "optimal" and r.gap_certified, r.status
    assert _close(r.objective, ref.objective), (r.objective, ref.objective)
    sense = 1.0 if seed == 73 else -1.0
    assert sense * (r.bound - r.objective) <= 1e-9, (r.bound, r.objective)


def _residual_lp():
    # min -x0  s.t.  x0 + y <= 1,  x0 in [0, 2],  y binary.  Optimum x0 = 1, y = 0.
    A_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([1.0])
    box = np.array([[0.0, 2.0], [0.0, 1.0]])
    c = np.array([-1.0, 0.0])
    return A_ub, b_ub, box, c


@pytest.mark.parametrize("excursion, verdict", [(1.2e-6, "repolished"), (1e-3, "not_snap_caused")])
def test_integral_claim_recovery_repolishes_only_lp_residuals(excursion, verdict):
    """The window is bounded: a node-LP residual is re-derived, a real excursion
    (the size the #952 regression tests inject) is left for the exit gate."""
    import time

    A_ub, b_ub, box, c = _residual_lp()
    x = np.array([1.0 + excursion, 0.0])

    def arbiter(p):
        return S._matrix_solution_feasible(np.asarray(p)[:2], A_ub, b_ub, None, None, box)

    assert not arbiter(x)
    got, point, obj = S._integral_claim_recovery(
        x,
        x.copy(),
        int_offsets=[1],
        int_sizes=[1],
        declared_box=box,
        c=c,
        obj_const=0.0,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=None,
        b_eq=None,
        t_start=time.perf_counter(),
        time_limit=30.0,
        arbiter=arbiter,
    )
    assert got == verdict
    if verdict == "repolished":
        assert arbiter(point)
        assert point[1] == 0.0
        assert obj == pytest.approx(-1.0, abs=1e-9)
    else:
        assert point is None and obj is None
