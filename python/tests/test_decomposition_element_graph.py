"""Element-level incidence graph and index coarsening (decomposition advisor).

The array-level graph (``ModelGraph``) treats every array variable as one
vertex, so a two-stage model whose scenarios share one recourse array reads as
a single block even after its first-stage integers are fixed. These tests pin
the scalar-level graph that sees through that, its labels, and the
coarsening that recovers the array-level view as a special case.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.advisor import StructureAnalyzer
from discopt.decomposition.graph.base import ModelGraph
from discopt.decomposition.graph.element import ElementGraph, ElementGraphError

S, T = 3, 4


def _two_stage_array_model():
    """Scenario recourse stored in one array ``x[s, t]``; ``build[s]`` binary.

    The budget row ties the binaries together, so the model as written is one
    block. Fixing the binaries splits the recourse into S independent scenario
    blocks, but every scenario's rows touch the same array ``x``.
    """
    m = dm.Model("two_stage_array")
    build = m.binary("build", shape=(S,))
    x = m.continuous("x", shape=(S, T), lb=0, ub=10)
    for s in range(S):
        for t in range(T):
            m.subject_to(x[s, t] <= 10 * build[s])
        for t in range(T - 1):
            m.subject_to(x[s, t + 1] - x[s, t] <= 2)
        m.subject_to(dm.sum(x[s, :]) >= 1)
    m.subject_to(dm.sum(build) <= S)
    m.minimize(dm.sum(build) + dm.sum(x))
    return m


def _vectorized_model():
    m = dm.Model("vectorized")
    cap = m.continuous("cap", lb=0, ub=10)
    x = m.continuous("x", shape=(S, T), lb=0, ub=10)
    m.subject_to(x <= cap, name="capc")
    m.subject_to(x[:, 1:] - x[:, :-1] <= 1, name="ramp")
    m.minimize(cap + dm.sum(x))
    return m


# ── the premise: array level hides the scenario split ──────────


def test_array_level_hides_block_diagonal_model():
    """Per-scenario arrays with no shared row: already S blocks as written."""
    m = dm.Model("diag")
    x = m.continuous("x", shape=(S, T), lb=0, ub=1)
    for s in range(S):
        m.subject_to(dm.sum(x[s, :]) <= 1)
    m.minimize(dm.sum(x))
    report = StructureAnalyzer().analyze(m)
    assert report.num_blocks == 1 and not report.is_block_diagonal
    assert report.element_blocks == S
    assert report.array_level_hides_blocks


def test_array_level_hides_scenario_split():
    m = _two_stage_array_model()
    report = StructureAnalyzer().analyze(m)
    # The array-level view: one vertex for ``x``, so fixing the integers
    # leaves one recourse block.
    assert report.blocks_after_integer_projection == 1
    assert not report.integer_localizes
    # The scalar-level view sees the S scenario blocks.
    assert report.element_graph_error is None
    assert report.element_blocks == 1
    assert report.element_blocks_after_integer_projection == S
    assert report.array_level_hides_blocks
    assert "array level hides blocks" in report.summary()


def test_no_false_hiding_flag_on_scalar_model():
    m = dm.Model("scalars")
    z = m.binary("z")
    a = m.continuous("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=5)
    m.subject_to(a <= 5 * z)
    m.subject_to(b <= 5 * z)
    m.minimize(a + b - z)
    report = StructureAnalyzer().analyze(m)
    assert report.blocks_after_integer_projection == 2
    assert report.element_blocks_after_integer_projection == 2
    assert not report.array_level_hides_blocks


def test_untouched_array_elements_are_not_blocks():
    """Elements no constraint touches must not count as independent blocks."""
    m = dm.Model("unused")
    x = m.continuous("x", lb=0, ub=1)
    m.continuous("spare", shape=(5,), lb=0, ub=1)
    m.subject_to(x <= 1)
    m.minimize(x)
    report = StructureAnalyzer().analyze(m)
    assert report.element_blocks == 1
    assert not report.array_level_hides_blocks


def test_model_without_objective_reports_why():
    """The array-level analysis works without an objective; so must the report."""
    m = dm.Model("feasibility")
    x = m.continuous("x", shape=(S,), lb=0, ub=1)
    m.subject_to(dm.sum(x) <= 1)
    report = StructureAnalyzer().analyze(m)
    assert report.num_blocks == 1
    assert report.element_blocks is None
    assert "no objective" in (report.element_graph_error or "")
    assert not report.array_level_hides_blocks
    assert "unavailable" in report.summary()


# ── labels and incidence ────────────────────────────────────────


def test_labels_and_incidence_two_stage():
    m = _two_stage_array_model()
    g = ElementGraph.from_model(m)
    assert g.checks.columns_labelled == S + S * T
    assert g.checks.rows_labelled == g.num_rows > 0
    assert g.checks.jacobian_entries == g.num_incidences > 0

    assert [g.col_label(j) for j in g.columns_of("build")] == [f"build[{s}]" for s in range(S)]
    xcols = g.columns_of("x")
    assert g.col_label(int(xcols[T + 2])) == "x[1, 2]"

    # Row 0 is ``x[0,0] <= 10*build[0]``: it touches exactly those two columns.
    assert set(g.row_cliques[0].tolist()) == {int(g.columns_of("build")[0]), int(xcols[0])}


def test_vectorized_constraint_rows_are_labelled_by_index():
    g = ElementGraph.from_model(_vectorized_model())
    assert g.con_names == ("capc", "ramp")
    assert g.con_shapes == ((S, T), (S, T - 1))
    assert g.row_label(0) == "capc[0, 0]"
    assert g.row_label(S * T - 1) == f"capc[{S - 1}, {T - 1}]"
    assert g.row_label(S * T) == "ramp[0, 0]"
    xcols = g.columns_of("x")
    # ramp[1, 2] is x[1,3] - x[1,2].
    i = S * T + 1 * (T - 1) + 2
    assert set(g.row_cliques[i].tolist()) == {int(xcols[T + 2]), int(xcols[T + 3])}
    # Dropping ``cap`` splits per scenario.
    assert g.bearing_blocks(drop_cols=g.columns_of("cap")) == S


def test_nonlinear_row_merges_blocks():
    m = _vectorized_model()
    x = m._variables[1]
    m.subject_to(x[0, 0] * x[1, 0] >= 1, name="nl")
    g = ElementGraph.from_model(m)
    block_of, _ = g.components(drop_cols=g.columns_of("cap"))
    xcols = g.columns_of("x").reshape(S, T)
    assert block_of[xcols[0, 0]] == block_of[xcols[1, 3]]
    assert block_of[xcols[0, 0]] != block_of[xcols[2, 0]]
    assert g.bearing_blocks(drop_cols=g.columns_of("cap")) == S - 1


def test_element_blocks_refine_array_blocks():
    """Every element block lies inside one array block (it can only split)."""
    rng = np.random.default_rng(1873)
    checked = 0
    for trial in range(12):
        m = dm.Model(f"rand{trial}")
        arrs = [m.continuous(f"v{k}", shape=(3, 2), lb=0, ub=1) for k in range(3)]
        for _ in range(int(rng.integers(2, 7))):
            k1, k2 = rng.integers(0, 3, size=2)
            i1, j1, i2, j2 = rng.integers(0, 3), rng.integers(0, 2), rng.integers(0, 3), 0
            m.subject_to(arrs[k1][i1, j1] + arrs[k2][i2, j2] <= 1)
        m.minimize(dm.sum(arrs[0]))
        g = ElementGraph.from_model(m)
        mg = ModelGraph.from_model(m)
        arr_block, _ = mg.variable_components()
        elem_block, _ = g.components()
        for b in np.unique(elem_block):
            owners = {arr_block[int(g.col_var[j])] for j in np.flatnonzero(elem_block == b)}
            assert len(owners) == 1
            checked += 1
    assert checked > 0


# ── coarsening ──────────────────────────────────────────────────


def test_coarsen_over_time_keeps_scenarios():
    g = ElementGraph.from_model(_two_stage_array_model())
    c = g.coarsen({"x": (1,)})
    labels = [c.col_label(s) for s in range(c.num_cols)]
    assert labels == [f"build[{s}]" for s in range(S)] + [f"x[{s}, :]" for s in range(S)]
    assert c.col_weight.tolist() == [1] * S + [T] * S
    assert int(c.col_weight.sum()) == g.num_cols
    assert int(c.row_weight.sum()) == g.num_rows
    # Same structure as the element graph: one block as written (the budget
    # row), S blocks once the binaries are removed.
    assert c.components()[1] == 1
    budget = c.num_rows - 1
    keep = [r for r in range(c.num_rows) if r != budget]
    sub = [[k for k in c.row_cliques[r].tolist() if k >= S] for r in keep]
    from discopt.decomposition.graph import kernels

    assert kernels.bearing_blocks(c.num_cols, sub) == S


def test_coarsen_all_axes_recovers_array_view():
    for m in (_two_stage_array_model(), _vectorized_model()):
        g = ElementGraph.from_model(m)
        var_axes = {n: tuple(range(len(sh))) for n, sh in zip(g.var_names, g.var_shapes)}
        con_axes = {n: tuple(range(len(sh))) for n, sh in zip(g.con_names, g.con_shapes)}
        c = g.coarsen(var_axes, con_axes)
        assert c.num_cols == len(g.var_names)
        assert c.components()[1] == ModelGraph.from_model(m).num_blocks()


def test_lift_maps_super_labels_to_columns():
    g = ElementGraph.from_model(_vectorized_model())
    c = g.coarsen({"x": (1,)})
    lifted = c.lift(np.arange(c.num_cols))
    xcols = g.columns_of("x").reshape(S, T)
    assert len(set(lifted[xcols[1]].tolist())) == 1
    assert lifted[xcols[0, 0]] != lifted[xcols[1, 0]]
    with pytest.raises(ValueError):
        c.lift(np.zeros(c.num_cols + 1))


def test_coarsen_rejects_unknown_names_and_axes():
    g = ElementGraph.from_model(_vectorized_model())
    with pytest.raises(KeyError):
        g.coarsen({"nope": (0,)})
    with pytest.raises(ValueError):
        g.coarsen({"x": (2,)})
    with pytest.raises(KeyError):
        g.coarsen(con_axes={"nope": (0,)})


# ── layout cross-checks ─────────────────────────────────────────


class _ShiftedEvaluator:
    """An evaluator whose layout disagrees with the model."""

    def __init__(self, real, *, extra_cols=0, drop_last_row=False):
        self._real = real
        self.n_variables = real.n_variables + extra_cols
        self.n_constraints = real.n_constraints
        self._drop = drop_last_row

    def constraint_row_map(self):
        rm = self._real.constraint_row_map()
        return rm[:-1] if self._drop else rm

    def jacobian_structure(self):
        return self._real.jacobian_structure()


def test_layout_mismatch_is_refused():
    from discopt._tape_nlp_evaluator import make_evaluator

    m = _vectorized_model()
    ev = make_evaluator(m)
    with pytest.raises(ElementGraphError, match="columns"):
        ElementGraph.from_model(m, _ShiftedEvaluator(ev, extra_cols=1))
    with pytest.raises(ElementGraphError, match="row map"):
        ElementGraph.from_model(m, _ShiftedEvaluator(ev, drop_last_row=True))


# ── Lagrangian block partition is element-level ─────────────────


def _shared_array_lagrangian():
    """S scenarios in one array ``x[s, t]``, joined only by the budget row."""
    m = dm.Model("shared_lag")
    x = m.continuous("x", shape=(S, T), lb=0, ub=10)
    z = m.integer("z", shape=(S,), lb=0, ub=5)
    for s in range(S):
        for t in range(T):
            m.subject_to(x[s, t] <= 2 * z[s])
        m.subject_to(dm.sum(x[s, :]) >= 3 + 2 * s)
    m.subject_to(dm.sum(z) <= 6, name="budget")
    m.mark_coupling("budget")
    m.minimize(dm.sum(x) + 3 * dm.sum(z))
    return m


@pytest.mark.requires_highs
def test_lagrangian_splits_scenarios_sharing_an_array(monkeypatch):
    import discopt.solvers.lp_backend as lp_backend
    from discopt.decomposition import solve_lagrangian

    mono = _shared_array_lagrangian().solve()
    assert mono.status == "optimal"

    real_get = lp_backend.get_decomposition_master_solver
    widths: list[int] = []

    def counting_get():
        milp, info = real_get()

        def wrapped(c, *args, **kwargs):
            widths.append(len(c))
            return milp(c, *args, **kwargs)

        return wrapped, info

    monkeypatch.setattr(lp_backend, "get_decomposition_master_solver", counting_get)
    result = solve_lagrangian(_shared_array_lagrangian(), max_iterations=60)

    # One block per scenario (T + 1 columns each), not one block of all of them.
    assert widths, "no block subproblem was solved"
    assert widths[:S] == [T + 1] * S
    assert max(widths) == T + 1
    assert result.bound <= mono.objective + 1e-6
    assert result.bound >= mono.objective - 1e-4


# ── element-level coupling scan ─────────────────────────────────


def test_separating_cliques_matches_brute_force_kernel():
    """The linear-time scan answers exactly what the per-clique kernel does."""
    from discopt.decomposition.graph import kernels
    from discopt.decomposition.graph.element import separating_cliques

    rng = np.random.default_rng(2611)
    compared = nonempty = 0
    for _ in range(3000):
        n = int(rng.integers(1, 10))
        m = int(rng.integers(0, 9))
        cliques = [
            sorted(set(rng.integers(0, n, size=int(rng.integers(0, 4))).tolist())) for _ in range(m)
        ]
        expected, truncated = kernels.bridge_cliques_status(cliques, n, 10**12)
        assert not truncated
        assert separating_cliques(n, cliques) == expected, (n, cliques)
        compared += 1
        nonempty += bool(expected)
    assert compared == 3000 and nonempty > 100


def test_budget_row_is_element_level_coupling():
    m = _two_stage_array_model()
    report = StructureAnalyzer().analyze(m)
    budget = len(m._constraints) - 1
    assert report.coupling_constraints == ()  # invisible at the array level
    assert report.element_coupling_constraints == (budget,)
    assert not report.coupling_scan_truncated


def test_non_row_constraints_couple_conservatively():
    """An SOS set or disjunction is not an evaluator row; it must still join
    the variables it references, or they would read as independent."""
    for add in (
        lambda m, x: m.sos1([x[0, 0], x[1, 0]]),
        lambda m, x: m.either_or([[x[0, 0] <= 0.2], [x[1, 0] >= 0.8]]),
    ):
        m = dm.Model("nonrow")
        x = m.continuous("x", shape=(2, 2), lb=0, ub=1)
        m.subject_to(x[0, 0] + x[0, 1] <= 1)
        m.subject_to(x[1, 0] + x[1, 1] <= 1)
        m.minimize(-dm.sum(x))
        assert ElementGraph.from_model(m).bearing_blocks() == 2
        add(m, x)
        g = ElementGraph.from_model(m)
        assert g.checks.pseudo_rows == 1
        assert g.bearing_blocks() == 1
        assert g.bridge_constraints() == []  # a pseudo-row is never offered


# ── advisor: element-level candidates ───────────────────────────


@pytest.mark.requires_highs
def test_advisor_recommends_benders_for_shared_array_two_stage():
    from discopt.decomposition import MethodKind, analyze_decomposition

    mono = _two_stage_array_model().solve()
    assert mono.status == "optimal"
    adv = analyze_decomposition(_two_stage_array_model())
    expl = adv.recommendation()
    assert expl.recommendation is MethodKind.BENDERS
    assert "element level" in adv.explain()
    decomposed = adv.decompose()
    assert decomposed.num_blocks == S
    assert decomposed.subproblems[1].variables == tuple(f"x[1, {t}]" for t in range(T))
    result = decomposed.solve()
    assert result.status == "optimal" and result.gap_certified
    assert result.objective == pytest.approx(mono.objective, abs=1e-6)


@pytest.mark.requires_highs
def test_element_lagrangian_candidate_bounds_the_optimum():
    from discopt.decomposition import MethodKind, analyze_decomposition

    mono = _shared_array_lagrangian().solve()
    model = _shared_array_lagrangian()
    model._coupling_keys = set()  # drop the annotation: the scan must find it
    adv = analyze_decomposition(model)
    lag = [c for c in adv.candidates() if c.method is MethodKind.LAGRANGIAN]
    assert len(lag) == 1 and lag[0].element_view is not None
    assert lag[0].num_blocks == S
    result = adv.decompose(MethodKind.LAGRANGIAN).solve(max_iterations=60)
    assert result.bound <= mono.objective + 1e-6


def test_element_generators_silent_where_array_view_sees_structure():
    """On a model whose arrays are per block, no element candidate appears."""
    from discopt.decomposition import analyze_decomposition

    m = dm.Model("per_block_arrays")
    z = m.binary("z")
    a = m.continuous("a", shape=(2,), lb=0, ub=5)
    b = m.continuous("b", shape=(2,), lb=0, ub=5)
    m.subject_to(a[0] + a[1] <= 5 * z)
    m.subject_to(b[0] + b[1] <= 5 * z)
    m.minimize(dm.sum(a) + dm.sum(b) - z)
    cands = analyze_decomposition(m).candidates()
    assert cands and all(c.element_view is None for c in cands)


# ── the advisor validates before reading the evaluator ─────────


_SHARED_SUM_ADVISOR = """
import resource, sys
# Linux only: macOS rejects any RLIMIT_AS cap ("current limit exceeds maximum
# limit"). Elsewhere the caller's 120 s timeout is the backstop.
if sys.platform.startswith("linux"):
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
import discopt.modeling as dm
from discopt.decomposition import analyze_decomposition

m = dm.Model("shared")
x = m.continuous("x", lb=0, ub=1)
y = m.continuous("y", lb=0, ub=1)
row = x - y
for _ in range(40):  # 41 nodes; 2**40 leaves as a tree
    row = row + row
m.subject_to(row <= 1.0)
m.minimize(x)
print(analyze_decomposition(m).structure().element_blocks)
"""


def test_advisor_on_unvalidated_shared_sum_does_not_blow_up():
    """A shared sum (``r = r + r`` repeated, #2100) is exponential as a tree.
    The element graph built the evaluator on the un-validated model, whose tape
    lowering expanded it until the process was killed for memory; the advisor
    now validates first, as every solve does. Run in a subprocess under a 3 GB
    cap so a regression fails this test instead of the test runner."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", _SHARED_SUM_ADVISOR],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "1"


_SHARED_SUM_DRIVER = """
import resource, sys
# Linux only: macOS rejects any RLIMIT_AS cap ("current limit exceeds maximum
# limit"). Elsewhere the caller's 120 s timeout is the backstop.
if sys.platform.startswith("linux"):
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
import discopt.modeling as dm
from discopt.decomposition import solve_benders, solve_lagrangian

m = dm.Model("shared_driver")
x = m.continuous("x", lb=0, ub=1)
y = m.continuous("y", lb=0, ub=1)
z = m.binary("z")
row = x - y
for _ in range(40):
    row = row + row
m.subject_to(row <= 1.0)
m.subject_to(x <= z)
m.subject_to(y <= 1 - z, name="c")
m.mark_coupling("c")
m.minimize(x + z)
print(getattr(sys.modules["__main__"], "DRIVER")(m).status)
"""


@pytest.mark.parametrize("driver", ["solve_benders", "solve_lagrangian"])
def test_drivers_on_unvalidated_shared_sum_return(driver):
    """Called directly, the drivers skipped Model.validate and their linear
    extraction walked a shared sum as a tree (a hang). Subprocess, 3 GB cap,
    120 s timeout, so a regression fails here instead of hanging the runner."""
    import subprocess
    import sys

    code = f"DRIVER = None\n{_SHARED_SUM_DRIVER}".replace(
        'getattr(sys.modules["__main__"], "DRIVER")', driver
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
