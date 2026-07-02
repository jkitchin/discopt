"""Tests for the optional Gurobi LP/MILP/QP/QCP backend."""

from __future__ import annotations

import sys
import types

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.solvers import MILPResult, QPResult, SolveStatus
from discopt.solvers import gurobi as gurobi_backend


def _install_fake_rust_classifier(monkeypatch, problem_kind: str) -> None:
    """Install just enough of discopt._rust for solver dispatch tests.

    The local source worktree used by these tests may not have a compiled PyO3
    extension, while CI builds it before running the broader suite.
    """

    linear_constraint_kinds = {"lp", "milp", "qp", "miqp"}
    quadratic_constraint_kinds = {"qcp", "qcqp", "miqcp", "miqcqp"}

    class _FakeRepr:
        n_constraints = 1 if problem_kind in quadratic_constraint_kinds else 0

        def is_objective_linear(self):
            return problem_kind in {"lp", "milp", "qcp", "miqcp"}

        def is_objective_quadratic(self):
            return problem_kind in {
                "lp",
                "milp",
                "qp",
                "miqp",
                "qcp",
                "miqcp",
                "qcqp",
                "miqcqp",
            }

        def is_constraint_linear(self, _idx):
            return problem_kind in linear_constraint_kinds

        def is_constraint_quadratic(self, _idx):
            return problem_kind in linear_constraint_kinds | quadratic_constraint_kinds

    fake_rust = types.SimpleNamespace(
        PyTreeManager=object,
        model_to_repr=lambda _model, _builder=None: _FakeRepr(),
    )
    monkeypatch.setitem(sys.modules, "discopt._rust", fake_rust)


def _require_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        env.dispose()
    except Exception as exc:
        pytest.skip(f"Gurobi is installed but no usable license is available: {exc}")


def test_gurobi_lp_validates_dimensions_without_importing_gurobipy():
    with pytest.raises(ValueError, match="columns"):
        gurobi_backend.solve_lp(
            c=np.array([1.0, 2.0]),
            A_ub=np.array([[1.0, 2.0, 3.0]]),
            b_ub=np.array([1.0]),
        )


def test_gurobi_lp_reports_missing_gurobipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "gurobipy", None)
    with pytest.raises(ImportError, match="gurobipy is required"):
        gurobi_backend.solve_lp(c=np.array([1.0]), bounds=[(0.0, 1.0)])


def test_gurobi_qp_validates_dimensions_without_importing_gurobipy():
    with pytest.raises(ValueError, match="Q has shape"):
        gurobi_backend.solve_qp(Q=np.eye(2), c=np.array([1.0]))


def test_gurobi_qp_reports_missing_gurobipy(monkeypatch):
    monkeypatch.setitem(sys.modules, "gurobipy", None)
    with pytest.raises(ImportError, match="gurobipy is required"):
        gurobi_backend.solve_qp(
            Q=np.array([[2.0]]),
            c=np.array([-2.0]),
            bounds=[(-5.0, 5.0)],
        )


def test_gurobi_qp_requires_explicit_nonconvex_option_without_importing_gurobipy():
    with pytest.raises(ValueError, match="gurobi_options.*options"):
        gurobi_backend.solve_qp(
            Q=np.array([[-2.0]]),
            c=np.array([0.0]),
            bounds=[(-1.0, 1.0)],
        )


def test_gurobi_qcp_validates_quadratic_constraint_dimensions_without_importing_gurobipy():
    with pytest.raises(ValueError, match="quadratic constraint 0 Q has shape"):
        gurobi_backend.solve_qcp(
            Q=np.zeros((2, 2)),
            c=np.array([1.0, 2.0]),
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            quadratic_constraints=[
                (np.eye(3), np.zeros(2), "<=", 1.0),
            ],
        )


def test_gurobi_qcp_requires_explicit_nonconvex_option_without_importing_gurobipy():
    with pytest.raises(ValueError, match="gurobi_options.*options"):
        gurobi_backend.solve_qcp(
            Q=np.zeros((1, 1)),
            c=np.array([0.0]),
            bounds=[(-2.0, 2.0)],
            quadratic_constraints=[
                (np.array([[-2.0]]), np.array([0.0]), "<=", 1.0),
            ],
        )


def test_milp_backend_selector_accepts_gurobi_without_importing_gurobipy(monkeypatch):
    from discopt.solvers.lp_backend import get_milp_solver

    monkeypatch.setitem(sys.modules, "gurobipy", None)

    solve_milp = get_milp_solver(backend="gurobi")

    assert solve_milp is gurobi_backend.solve_milp
    with pytest.raises(ImportError, match="gurobipy is required"):
        solve_milp(
            c=np.array([1.0]),
            bounds=[(0.0, 1.0)],
            integrality=np.array([1]),
        )


def test_milp_backend_selector_lists_gurobi_in_invalid_backend_error():
    from discopt.solvers.lp_backend import get_milp_solver

    with pytest.raises(ValueError, match="gurobi"):
        get_milp_solver(backend="not-a-backend")


def test_set_common_params_keeps_explicit_threads_over_options():
    class Recorder:
        def __init__(self):
            self.params = {}

        def setParam(self, key, value):
            self.params[key] = value

    model = Recorder()

    gurobi_backend._set_common_params(
        model,
        time_limit=7.0,
        threads=2,
        options={"TimeLimit": 3.0, "Threads": 8},
    )

    assert model.params["TimeLimit"] == 7.0
    assert model.params["Threads"] == 2


def test_gurobi_milp_optimal_result_reports_zero_gap(monkeypatch):
    class FakeGRB:
        CONTINUOUS = "C"
        BINARY = "B"
        INTEGER = "I"
        MINIMIZE = 1
        INFINITY = 1e100
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        INF_OR_UNBD = 4
        ITERATION_LIMIT = 7
        TIME_LIMIT = 9

    class FakeEnv:
        def __init__(self, empty=True):
            self.empty = empty

        def setParam(self, *_args):
            pass

        def start(self):
            pass

        def dispose(self):
            pass

    class FakeMVar:
        __array_priority__ = 1000
        X = np.array([1.0])

        def __rmatmul__(self, _other):
            return 0.0

    class FakeModel:
        Status = FakeGRB.OPTIMAL
        NodeCount = 0
        Runtime = 0.0
        SolCount = 1
        ObjVal = 0.94
        ObjBound = 0.940002
        MIPGap = 0.25

        def __init__(self, *_args, **_kwargs):
            pass

        def setParam(self, *_args):
            pass

        def addMVar(self, **_kwargs):
            return FakeMVar()

        def setObjective(self, *_args):
            pass

        def optimize(self):
            pass

        def dispose(self):
            pass

    fake_gp = types.SimpleNamespace(Env=FakeEnv, Model=FakeModel)
    monkeypatch.setattr(gurobi_backend, "_load_gurobi", lambda: (fake_gp, FakeGRB))

    result = gurobi_backend.solve_milp(
        c=np.array([1.0]),
        bounds=[(0.0, 1.0)],
        integrality=np.array([1], dtype=np.int32),
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.objective == pytest.approx(0.94)
    assert result.bound == pytest.approx(result.objective)
    assert result.gap == 0.0


def test_gurobi_milp_solution_pool_sets_params_and_returns_candidates(monkeypatch):
    class FakeGRB:
        CONTINUOUS = "C"
        BINARY = "B"
        INTEGER = "I"
        MINIMIZE = 1
        INFINITY = 1e100
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        INF_OR_UNBD = 4
        ITERATION_LIMIT = 7
        TIME_LIMIT = 9

    class FakeEnv:
        def __init__(self, empty=True):
            self.empty = empty

        def setParam(self, *_args):
            pass

        def start(self):
            pass

        def dispose(self):
            pass

    class FakeMVar:
        __array_priority__ = 1000

        def __init__(self, model):
            self.model = model

        @property
        def X(self):
            return self.model.solutions[0]

        @property
        def Xn(self):
            return self.model.solutions[self.model.solution_number]

        def __rmatmul__(self, _other):
            return 0.0

    class FakeModel:
        Status = FakeGRB.OPTIMAL
        NodeCount = 0
        Runtime = 0.0
        SolCount = 3
        ObjVal = 1.0
        ObjBound = 1.0
        MIPGap = 0.0
        last = None

        def __init__(self, *_args, **_kwargs):
            self.params = {}
            self.solution_number = 0
            self.solutions = [
                np.array([0.0]),
                np.array([1.0]),
                np.array([0.5]),
            ]
            self.pool_objectives = [1.0, 3.0, 2.0]
            FakeModel.last = self

        @property
        def PoolObjVal(self):
            return self.pool_objectives[self.solution_number]

        def setParam(self, key, value):
            self.params[key] = value
            if key == "SolutionNumber":
                self.solution_number = int(value)

        def addMVar(self, **_kwargs):
            return FakeMVar(self)

        def setObjective(self, *_args):
            pass

        def optimize(self):
            pass

        def dispose(self):
            pass

    fake_gp = types.SimpleNamespace(Env=FakeEnv, Model=FakeModel)
    monkeypatch.setattr(gurobi_backend, "_load_gurobi", lambda: (fake_gp, FakeGRB))

    result = gurobi_backend.solve_milp(
        c=np.array([1.0]),
        bounds=[(0.0, 1.0)],
        integrality=np.array([1], dtype=np.int32),
        solution_pool=True,
        num_solution_iteration=3,
    )

    assert FakeModel.last.params["PoolSearchMode"] == 2
    assert FakeModel.last.params["PoolSolutions"] == 3
    assert result.solution_pool_objectives == pytest.approx([1.0, 2.0, 3.0])
    assert [candidate.tolist() for candidate in result.solution_pool] == [[0.0], [0.5], [1.0]]


def test_gurobi_milp_callbacks_add_lazy_and_node_cuts(monkeypatch):
    class FakeCallback:
        MIPSOL = 1
        MIPNODE = 2
        MIPNODE_STATUS = 3

    class FakeGRB:
        CONTINUOUS = "C"
        BINARY = "B"
        INTEGER = "I"
        MINIMIZE = 1
        INFINITY = 1e100
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        INF_OR_UNBD = 4
        ITERATION_LIMIT = 7
        TIME_LIMIT = 9
        Callback = FakeCallback

    class FakeEnv:
        def __init__(self, empty=True):
            self.empty = empty

        def setParam(self, *_args):
            pass

        def start(self):
            pass

        def dispose(self):
            pass

    class FakeMVar:
        __array_priority__ = 1000
        X = np.array([0.0])

        def __rmatmul__(self, _other):
            return 0.0

    class FakeModel:
        Status = FakeGRB.OPTIMAL
        NodeCount = 1
        Runtime = 0.0
        SolCount = 1
        ObjVal = 0.0
        ObjBound = 0.0
        MIPGap = 0.0
        last = None

        def __init__(self, *_args, **_kwargs):
            self.params = {}
            self.lazy_constraints = []
            self.node_cuts = []
            FakeModel.last = self

        def setParam(self, key, value):
            self.params[key] = value

        def addMVar(self, **_kwargs):
            return FakeMVar()

        def setObjective(self, *_args):
            pass

        def cbGetSolution(self, _x):
            return np.array([0.0])

        def cbGet(self, attr):
            assert attr == FakeGRB.Callback.MIPNODE_STATUS
            return FakeGRB.OPTIMAL

        def cbGetNodeRel(self, _x):
            return np.array([0.25])

        def cbLazy(self, expr):
            self.lazy_constraints.append(expr)

        def cbCut(self, expr):
            self.node_cuts.append(expr)

        def optimize(self, callback=None):
            if callback is not None:
                callback(self, FakeGRB.Callback.MIPNODE)
                callback(self, FakeGRB.Callback.MIPSOL)

        def dispose(self):
            pass

    fake_gp = types.SimpleNamespace(Env=FakeEnv, Model=FakeModel)
    monkeypatch.setattr(gurobi_backend, "_load_gurobi", lambda: (fake_gp, FakeGRB))

    callback_candidates = []

    def lazy_callback(candidate):
        callback_candidates.append(candidate)
        return [(np.array([1.0]), -0.5)]

    node_candidates = []

    def node_callback(candidate):
        node_candidates.append(candidate)
        return [(np.array([1.0]), 0.0)]

    result = gurobi_backend.solve_milp_with_lazy_cuts(
        c=np.array([1.0]),
        bounds=[(0.0, 1.0)],
        integrality=np.array([1], dtype=np.int32),
        lazy_callback=lazy_callback,
        node_callback=node_callback,
    )

    assert result.status == SolveStatus.OPTIMAL
    assert callback_candidates
    assert node_candidates
    assert FakeModel.last.params["LazyConstraints"] == 1
    assert FakeModel.last.params["PreCrush"] == 1
    assert len(FakeModel.last.lazy_constraints) == 1
    assert len(FakeModel.last.node_cuts) == 1
    assert result.callback_stats["mipsol_calls"] == 1
    assert result.callback_stats["mipnode_calls"] == 1
    assert result.callback_stats["lazy_cuts"] == 1
    assert result.callback_stats["node_cuts"] == 1


def test_amp_bound_tolerance_closes_small_master_bound_inversion():
    from discopt.solvers import amp as amp_module

    abs_gap, order_ok = amp_module._amp_abs_gap_with_bound_tolerance(
        lower_bound=0.940002,
        upper_bound=0.94,
        abs_tol=1e-6,
    )

    assert order_ok is True
    assert abs_gap == 0.0


def test_model_solve_amp_forwards_gurobi_milp_solver(monkeypatch):
    import discopt.solvers.amp as amp_module

    calls = {}

    def fake_solve_amp(model, **kwargs):
        calls["model"] = model
        calls["milp_solver"] = kwargs["milp_solver"]
        calls["rel_gap"] = kwargs["rel_gap"]
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(amp_module, "solve_amp", fake_solve_amp)

    m = dm.Model("amp_gurobi_milp_solver_forwarding")
    x = m.binary("x")
    m.minimize(x)

    result = m.solve(solver="amp", milp_solver="gurobi", rel_gap=1e-5)

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["milp_solver"] == "gurobi"
    assert calls["rel_gap"] == 1e-5


def test_model_solve_oa_forwards_gurobi_milp_solver(monkeypatch):
    import discopt.solvers.mip_nlp as mip_nlp_module

    calls = {}

    def fake_solve_mip_nlp(model, **kwargs):
        calls["model"] = model
        calls["method"] = kwargs["method"]
        calls["milp_solver"] = kwargs["milp_solver"]
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(mip_nlp_module, "solve_mip_nlp", fake_solve_mip_nlp)

    m = dm.Model("oa_gurobi_milp_solver_forwarding")
    x = m.binary("x")
    m.minimize(x)

    result = m.solve(solver="mip-nlp", mip_nlp_method="oa", milp_solver="gurobi")

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["method"] == "oa"
    assert calls["milp_solver"] == "gurobi"


def test_model_solve_loa_forwards_gurobi_milp_solver(monkeypatch):
    import discopt.solvers.gdpopt_loa as loa_module

    calls = {}

    def fake_solve_gdpopt_loa(model, **kwargs):
        calls["model"] = model
        calls["milp_solver"] = kwargs["milp_solver"]
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(loa_module, "solve_gdpopt_loa", fake_solve_gdpopt_loa)

    m = dm.Model("loa_gurobi_milp_solver_forwarding")
    x = m.binary("x")
    m.minimize(x)

    result = m.solve(gdp_method="loa", milp_solver="gurobi", skip_convex_check=True)

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["milp_solver"] == "gurobi"


@pytest.mark.parametrize(
    ("problem_kind", "make_var"),
    [
        ("nlp", lambda m: m.continuous("x", lb=-2, ub=2)),
        ("minlp", lambda m: m.binary("x")),
    ],
)
def test_model_solve_gurobi_rejects_general_nonlinear_classes(monkeypatch, problem_kind, make_var):
    _install_fake_rust_classifier(monkeypatch, problem_kind)

    m = dm.Model(f"gurobi_rejects_{problem_kind}")
    x = make_var(m)
    m.minimize(x)

    with pytest.raises(NotImplementedError, match=f"classified this model as {problem_kind!r}"):
        m.solve(solver="gurobi")


def test_model_solve_gurobi_dispatches_lp(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "lp")
    import discopt.solver as solver

    calls = {}

    def fake_gurobi_lp(model, t_start, time_limit=None, threads=None, options=None):
        calls["model"] = model
        calls["time_limit"] = time_limit
        calls["threads"] = threads
        calls["options"] = options
        return SolveResult(status="optimal", objective=1.0, bound=1.0, gap=0.0)

    monkeypatch.setattr(solver, "_solve_lp_gurobi", fake_gurobi_lp)

    m = dm.Model("lp_dispatch_gurobi")
    x = m.continuous("x", lb=0, ub=5)
    m.minimize(x + 1)

    result = m.solve(
        solver="gurobi",
        time_limit=12.0,
        threads=3,
        gurobi_options={"Method": 1},
    )

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["time_limit"] == 12.0
    assert calls["threads"] == 3
    assert calls["options"] == {"Method": 1}


def test_model_solve_gurobi_dispatches_milp(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "milp")
    import discopt.solver as solver

    calls = {}

    def fake_gurobi_milp(
        model,
        t_start,
        time_limit=None,
        gap_tolerance=1e-4,
        threads=None,
        options=None,
    ):
        calls["model"] = model
        calls["gap_tolerance"] = gap_tolerance
        calls["threads"] = threads
        calls["options"] = options
        return SolveResult(status="optimal", objective=2.0, bound=2.0, gap=0.0)

    monkeypatch.setattr(solver, "_solve_milp_gurobi", fake_gurobi_milp)

    m = dm.Model("milp_dispatch_gurobi")
    y = m.integer("y", lb=0, ub=2)
    m.minimize(y)

    result = m.solve(
        solver="gurobi",
        gap_tolerance=1e-5,
        threads=2,
        gurobi_options={"MIPFocus": 1},
    )

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["gap_tolerance"] == 1e-5
    assert calls["threads"] == 2
    assert calls["options"] == {"MIPFocus": 1}


def test_model_solve_gurobi_dispatches_qp(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "qp")
    import discopt.solver as solver

    calls = {}

    def fake_gurobi_qp(
        model,
        t_start,
        time_limit=None,
        gap_tolerance=1e-4,
        threads=None,
        options=None,
    ):
        calls["model"] = model
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        calls["threads"] = threads
        calls["options"] = options
        return SolveResult(status="optimal", objective=3.0, bound=3.0, gap=0.0)

    monkeypatch.setattr(solver, "_solve_qp_gurobi", fake_gurobi_qp)

    m = dm.Model("qp_dispatch_gurobi")
    x = m.continuous("x", lb=0, ub=2)
    m.minimize(x**2)

    result = m.solve(
        solver="gurobi",
        time_limit=9.0,
        gap_tolerance=1e-6,
        threads=4,
        gurobi_options={"Method": 2},
    )

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["time_limit"] == 9.0
    assert calls["gap_tolerance"] == 1e-6
    assert calls["threads"] == 4
    assert calls["options"] == {"Method": 2}


def test_model_solve_gurobi_dispatches_miqp(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "miqp")
    import discopt.solver as solver

    calls = {}

    def fake_gurobi_qp(
        model,
        t_start,
        time_limit=None,
        gap_tolerance=1e-4,
        threads=None,
        options=None,
    ):
        calls["model"] = model
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        calls["threads"] = threads
        calls["options"] = options
        return SolveResult(status="optimal", objective=4.0, bound=4.0, gap=0.0)

    monkeypatch.setattr(solver, "_solve_qp_gurobi", fake_gurobi_qp)

    m = dm.Model("miqp_dispatch_gurobi")
    y = m.integer("y", lb=0, ub=2)
    m.minimize((y - 1) ** 2)

    result = m.solve(
        solver="gurobi",
        time_limit=8.0,
        gap_tolerance=1e-5,
        threads=2,
        gurobi_options={"MIPFocus": 1},
    )

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["time_limit"] == 8.0
    assert calls["gap_tolerance"] == 1e-5
    assert calls["threads"] == 2
    assert calls["options"] == {"MIPFocus": 1}


def test_model_solve_gurobi_dispatches_qcp(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "qcp")
    import discopt.solver as solver

    calls = {}

    def fake_gurobi_qcp(
        model,
        t_start,
        time_limit=None,
        gap_tolerance=1e-4,
        threads=None,
        options=None,
    ):
        calls["model"] = model
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        calls["threads"] = threads
        calls["options"] = options
        return SolveResult(status="optimal", objective=-1.0, bound=-1.0, gap=0.0)

    monkeypatch.setattr(solver, "_solve_qcp_gurobi", fake_gurobi_qcp)

    m = dm.Model("qcp_dispatch_gurobi")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(x)
    m.subject_to(x**2 <= 1)

    result = m.solve(
        solver="gurobi",
        time_limit=7.0,
        gap_tolerance=1e-6,
        threads=2,
        gurobi_options={"BarConvTol": 1e-8},
    )

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["time_limit"] == 7.0
    assert calls["gap_tolerance"] == 1e-6
    assert calls["threads"] == 2
    assert calls["options"] == {"BarConvTol": 1e-8}


def test_model_solve_gurobi_dispatches_miqcqp(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "miqcqp")
    import discopt.solver as solver

    calls = {}

    def fake_gurobi_qcp(
        model,
        t_start,
        time_limit=None,
        gap_tolerance=1e-4,
        threads=None,
        options=None,
    ):
        calls["model"] = model
        calls["time_limit"] = time_limit
        calls["gap_tolerance"] = gap_tolerance
        calls["threads"] = threads
        calls["options"] = options
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(solver, "_solve_qcp_gurobi", fake_gurobi_qcp)

    m = dm.Model("miqcqp_dispatch_gurobi")
    x = m.binary("x")
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize((y - 1) ** 2 + x)
    m.subject_to(y**2 <= x)

    result = m.solve(
        solver="gurobi",
        time_limit=6.0,
        gap_tolerance=1e-5,
        threads=1,
        gurobi_options={"MIPFocus": 1},
    )

    assert result.status == "optimal"
    assert calls["model"] is m
    assert calls["time_limit"] == 6.0
    assert calls["gap_tolerance"] == 1e-5
    assert calls["threads"] == 1
    assert calls["options"] == {"MIPFocus": 1}


def test_gurobi_milp_maximize_time_limit_maps_dual_bound(monkeypatch):
    _install_fake_rust_classifier(monkeypatch, "milp")
    import discopt.solver as solver
    from discopt._jax import problem_classifier

    m = dm.Model("gurobi_max_milp_bound_mapping")
    y = m.integer("y", lb=0, ub=10)
    m.maximize(y)

    lp_data = problem_classifier.LPData(
        c=np.array([-1.0]),
        A_eq=np.zeros((0, 1)),
        b_eq=np.zeros(0),
        x_l=np.array([0.0]),
        x_u=np.array([10.0]),
        obj_const=0.0,
    )
    monkeypatch.setattr(problem_classifier, "extract_lp_data", lambda _model: lp_data)

    def fake_solve_milp(**_kwargs):
        return MILPResult(
            status=SolveStatus.TIME_LIMIT,
            x=np.array([8.0]),
            objective=-8.0,
            bound=-10.0,
            gap=0.25,
            node_count=7,
        )

    monkeypatch.setattr(gurobi_backend, "solve_milp", fake_solve_milp)

    result = solver._solve_milp_gurobi(m, t_start=0.0, time_limit=1.0)

    assert result.status == "time_limit"
    assert result.objective == pytest.approx(8.0)
    assert result.bound == pytest.approx(10.0)
    assert result.gap == pytest.approx(0.25)
    assert result.node_count == 7


def test_gurobi_qp_maximize_constant_maps_objective(monkeypatch):
    import discopt.solver as solver
    from discopt._jax import problem_classifier

    m = dm.Model("gurobi_max_qp_objective_mapping")
    x = m.continuous("x", lb=0, ub=2)
    m.maximize(7 - (x - 1) ** 2)

    qp_data = problem_classifier.QPData(
        Q=np.array([[2.0]]),
        c=np.array([-2.0]),
        A_eq=np.zeros((0, 1)),
        b_eq=np.zeros(0),
        x_l=np.array([0.0]),
        x_u=np.array([2.0]),
        obj_const=-6.0,
    )
    monkeypatch.setattr(problem_classifier, "extract_qp_data", lambda _model: qp_data)

    def fake_solve_qp(**kwargs):
        assert kwargs["gap_tolerance"] == 1e-4
        np.testing.assert_allclose(kwargs["Q"], [[2.0]])
        return QPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([1.0]),
            objective=-1.0,
            bound=-1.0,
            gap=0.0,
        )

    monkeypatch.setattr(gurobi_backend, "solve_qp", fake_solve_qp)

    result = solver._solve_qp_gurobi(m, t_start=0.0)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(7.0)
    assert result.bound == pytest.approx(7.0)
    assert result.gap == pytest.approx(0.0)
    assert result.x["x"] == pytest.approx(1.0)


def test_gurobi_miqp_time_limit_maps_incumbent_bound_and_gap(monkeypatch):
    import discopt.solver as solver
    from discopt._jax import problem_classifier

    m = dm.Model("gurobi_max_miqp_bound_mapping")
    y = m.integer("y", lb=0, ub=2)
    m.maximize(7 - (y - 1) ** 2)

    qp_data = problem_classifier.QPData(
        Q=np.array([[2.0]]),
        c=np.array([-2.0]),
        A_eq=np.zeros((0, 1)),
        b_eq=np.zeros(0),
        x_l=np.array([0.0]),
        x_u=np.array([2.0]),
        obj_const=-6.0,
    )
    monkeypatch.setattr(problem_classifier, "extract_qp_data", lambda _model: qp_data)

    def fake_solve_qp(**_kwargs):
        return QPResult(
            status=SolveStatus.TIME_LIMIT,
            x=np.array([1.0]),
            objective=-1.0,
            bound=-0.5,
            gap=0.25,
            node_count=5,
        )

    monkeypatch.setattr(gurobi_backend, "solve_qp", fake_solve_qp)

    result = solver._solve_qp_gurobi(m, t_start=0.0, time_limit=1.0)

    assert result.status == "time_limit"
    assert result.objective == pytest.approx(7.0)
    assert result.bound == pytest.approx(6.5)
    assert result.gap == pytest.approx(0.5 / 7.0)
    assert result.node_count == 5
    assert result.x["y"] == pytest.approx(1.0)


def test_gurobi_qcp_maximize_maps_objective(monkeypatch):
    import discopt.solver as solver
    from discopt._jax import problem_classifier

    m = dm.Model("gurobi_max_qcp_objective_mapping")
    x = m.continuous("x", lb=-2, ub=2)
    m.maximize(x)
    m.subject_to(x**2 <= 1)

    qcp_data = problem_classifier.QCPData(
        Q=np.zeros((1, 1)),
        c=np.array([-1.0]),
        A_ub=np.zeros((0, 1)),
        b_ub=np.zeros(0),
        A_eq=np.zeros((0, 1)),
        b_eq=np.zeros(0),
        quadratic_constraints=(
            problem_classifier.QuadraticConstraintData(
                Q=np.array([[2.0]]),
                c=np.array([0.0]),
                sense="<=",
                rhs=1.0,
            ),
        ),
        x_l=np.array([-2.0]),
        x_u=np.array([2.0]),
        obj_const=0.0,
    )
    monkeypatch.setattr(problem_classifier, "extract_qcp_data", lambda _model: qcp_data)

    def fake_solve_qcp(**kwargs):
        assert len(kwargs["quadratic_constraints"]) == 1
        return QPResult(
            status=SolveStatus.OPTIMAL,
            x=np.array([1.0]),
            objective=-1.0,
            bound=-1.0,
            gap=0.0,
        )

    monkeypatch.setattr(gurobi_backend, "solve_qcp", fake_solve_qcp)

    result = solver._solve_qcp_gurobi(m, t_start=0.0)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(1.0)
    assert result.bound == pytest.approx(1.0)
    assert result.x["x"] == pytest.approx(1.0)


def test_gurobi_qcp_maximize_time_limit_maps_incumbent_bound_and_gap(monkeypatch):
    import discopt.solver as solver
    from discopt._jax import problem_classifier

    m = dm.Model("gurobi_max_qcp_time_limit_mapping")
    x = m.continuous("x", lb=-2, ub=2)
    m.maximize(x)
    m.subject_to(x**2 <= 1)

    qcp_data = problem_classifier.QCPData(
        Q=np.zeros((1, 1)),
        c=np.array([-1.0]),
        A_ub=np.zeros((0, 1)),
        b_ub=np.zeros(0),
        A_eq=np.zeros((0, 1)),
        b_eq=np.zeros(0),
        quadratic_constraints=(
            problem_classifier.QuadraticConstraintData(
                Q=np.array([[2.0]]),
                c=np.array([0.0]),
                sense="<=",
                rhs=1.0,
            ),
        ),
        x_l=np.array([-2.0]),
        x_u=np.array([2.0]),
        obj_const=0.0,
    )
    monkeypatch.setattr(problem_classifier, "extract_qcp_data", lambda _model: qcp_data)

    def fake_solve_qcp(**_kwargs):
        return QPResult(
            status=SolveStatus.TIME_LIMIT,
            x=np.array([0.75]),
            objective=-0.75,
            bound=-1.0,
            gap=0.2,
            node_count=9,
        )

    monkeypatch.setattr(gurobi_backend, "solve_qcp", fake_solve_qcp)

    result = solver._solve_qcp_gurobi(m, t_start=0.0, time_limit=1.0)

    assert result.status == "time_limit"
    assert result.objective == pytest.approx(0.75)
    assert result.bound == pytest.approx(1.0)
    assert result.gap == pytest.approx(0.25)
    assert result.node_count == 9
    assert result.x["x"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("backend_status", "public_status"),
    [
        (SolveStatus.INFEASIBLE, "infeasible"),
        (SolveStatus.UNBOUNDED, "unbounded"),
        (SolveStatus.ITERATION_LIMIT, "iteration_limit"),
        (SolveStatus.ERROR, "error"),
    ],
)
def test_gurobi_qcp_no_incumbent_status_mapping(monkeypatch, backend_status, public_status):
    import discopt.solver as solver
    from discopt._jax import problem_classifier

    m = dm.Model("gurobi_qcp_no_incumbent_status_mapping")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(x)
    m.subject_to(x**2 <= 1)

    qcp_data = problem_classifier.QCPData(
        Q=np.zeros((1, 1)),
        c=np.array([1.0]),
        A_ub=np.zeros((0, 1)),
        b_ub=np.zeros(0),
        A_eq=np.zeros((0, 1)),
        b_eq=np.zeros(0),
        quadratic_constraints=(
            problem_classifier.QuadraticConstraintData(
                Q=np.array([[2.0]]),
                c=np.array([0.0]),
                sense="<=",
                rhs=1.0,
            ),
        ),
        x_l=np.array([-2.0]),
        x_u=np.array([2.0]),
        obj_const=0.0,
    )
    monkeypatch.setattr(problem_classifier, "extract_qcp_data", lambda _model: qcp_data)

    def fake_solve_qcp(**_kwargs):
        return QPResult(status=backend_status, node_count=6)

    monkeypatch.setattr(gurobi_backend, "solve_qcp", fake_solve_qcp)

    result = solver._solve_qcp_gurobi(m, t_start=0.0, time_limit=1.0)

    assert result.status == public_status
    assert result.objective is None
    assert result.bound is None
    assert result.gap is None
    assert result.x is None
    assert result.node_count == 6


def test_gurobi_lp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_lp(
        c=np.array([-1.0, -2.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([10.0]),
        bounds=[(0.0, float("inf")), (0.0, float("inf"))],
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    np.testing.assert_allclose(result.x, [0.0, 10.0], atol=1e-6)
    assert result.objective == pytest.approx(-20.0)


def test_gurobi_milp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_milp(
        c=np.array([-1.0, -2.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([4.0]),
        bounds=[(0.0, 4.0), (0.0, 4.0)],
        integrality=np.array([0, 1], dtype=np.int32),
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    np.testing.assert_allclose(result.x, [0.0, 4.0], atol=1e-6)
    assert result.objective == pytest.approx(-8.0)


def test_gurobi_qp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_qp(
        Q=np.array([[2.0, 0.0], [0.0, 2.0]]),
        c=np.array([-2.0, -4.0]),
        A_eq=np.array([[1.0, 1.0]]),
        b_eq=np.array([3.0]),
        bounds=[(0.0, float("inf")), (0.0, float("inf"))],
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-6)
    assert result.objective == pytest.approx(-5.0)


def test_gurobi_miqp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_qp(
        Q=np.array([[2.0, 0.0], [0.0, 2.0]]),
        c=np.array([-2.0, -4.0]),
        bounds=[(0.0, 1.0), (0.0, 3.0)],
        integrality=np.array([1, 0], dtype=np.int32),
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-6)
    assert result.objective == pytest.approx(-5.0)


def test_gurobi_qcp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_qcp(
        Q=np.zeros((1, 1)),
        c=np.array([1.0]),
        bounds=[(-2.0, 2.0)],
        quadratic_constraints=[
            (np.array([[2.0]]), np.array([0.0]), "<=", 1.0),
        ],
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    assert result.x[0] == pytest.approx(-1.0, abs=1e-6)
    assert result.objective == pytest.approx(-1.0, abs=1e-6)


def test_gurobi_miqcp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_qcp(
        Q=np.zeros((2, 2)),
        c=np.array([-1.0, 0.0]),
        bounds=[(0.0, 2.0), (0.0, 1.0)],
        quadratic_constraints=[
            (np.array([[2.0, 0.0], [0.0, 0.0]]), np.array([0.0, -1.0]), "<=", 0.0),
        ],
        integrality=np.array([0, 1], dtype=np.int32),
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-6)
    assert result.objective == pytest.approx(-1.0, abs=1e-6)


def test_gurobi_nonconvex_qcp_smoke_if_available():
    _require_gurobi()

    result = gurobi_backend.solve_qcp(
        Q=np.zeros((1, 1)),
        c=np.array([1.0]),
        bounds=[(-2.0, 2.0)],
        quadratic_constraints=[
            (np.array([[2.0]]), np.array([0.0]), ">=", 1.0),
        ],
        options={"NonConvex": 2},
    )

    assert result.status == SolveStatus.OPTIMAL
    assert result.x is not None
    assert result.x[0] == pytest.approx(-2.0, abs=1e-6)
    assert result.objective == pytest.approx(-2.0, abs=1e-6)


def test_model_solve_gurobi_qp_smoke_if_available():
    _require_gurobi()

    m = dm.Model("gurobi_qp_model_solve_smoke")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.subject_to(x + y == 3)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)

    result = m.solve(solver="gurobi", time_limit=30.0)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-7)
    assert result.x["x"] == pytest.approx(1.0, abs=1e-6)
    assert result.x["y"] == pytest.approx(2.0, abs=1e-6)


def test_model_solve_gurobi_miqp_smoke_if_available():
    _require_gurobi()

    m = dm.Model("gurobi_miqp_model_solve_smoke")
    x = m.binary("x")
    y = m.continuous("y", lb=0, ub=3)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)

    result = m.solve(solver="gurobi", time_limit=30.0)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-7)
    assert result.x["x"] == pytest.approx(1.0, abs=1e-6)
    assert result.x["y"] == pytest.approx(2.0, abs=1e-6)


def test_model_solve_gurobi_qcp_smoke_if_available():
    _require_gurobi()

    m = dm.Model("gurobi_qcp_model_solve_smoke")
    x = m.continuous("x", lb=-2, ub=2)
    m.minimize(x)
    m.subject_to(x**2 <= 1)

    result = m.solve(solver="gurobi", time_limit=30.0)

    assert result.status == "optimal"
    assert result.objective == pytest.approx(-1.0, abs=1e-6)
    assert result.x["x"] == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.slow
def test_oa_gurobi_converges_on_degenerate_convex_minlp_if_available():
    _require_gurobi()

    m = dm.Model("oa_gurobi_degenerate_minlp")
    x = m.integer("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.subject_to(x + y >= 3)
    m.minimize((x - 2) ** 2 + (y - 1.5) ** 2)

    result = m.solve(
        solver="mip-nlp",
        mip_nlp_method="oa",
        milp_solver="gurobi",
        max_nodes=20,
        time_limit=30,
    )

    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-6)
    assert result.gap is not None
    assert result.gap <= 1e-6


@pytest.mark.slow
def test_amp_gurobi_certifies_convex_minlp_if_available():
    _require_gurobi()

    m = dm.Model("amp_gurobi_bound_tolerance")
    x = m.integer("x", lb=0, ub=6)
    y = m.continuous("y", lb=0, ub=6)
    m.subject_to(x + y >= 4)
    m.minimize((x - 3.2) ** 2 + 2 * (y - 2.1) ** 2 + 0.3 * x)

    result = m.solve(solver="amp", milp_solver="gurobi", rel_gap=1e-6, time_limit=30)

    assert result.status == "optimal"
    assert result.gap_certified is True
    assert result.objective == pytest.approx(0.94, abs=1e-5)
    assert result.bound is not None
    assert result.gap is not None
    assert result.gap <= 1e-6
