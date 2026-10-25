"""#1516: DAE models must convert to the Rust ``ModelRepr``.

``DAEBuilder`` broadcasts a piecewise-constant control over the collocation
points with ``u[:, None]`` (numpy's ``np.newaxis``). ``model_to_repr`` could not
represent a ``None`` subscript and raised ``TypeError: 'NoneType' object cannot
be interpreted as an integer``. Every caller caught that and carried on, so on
every collocation model with a control the whole Rust layer -- root presolve,
FBBT, in-tree presolve, the problem classifier, the arena tape -- silently never
ran. ``IndexElem::NewAxis`` now represents it exactly, and ``solve_model`` logs a
repr-build failure as a WARNING instead of a debug line.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import model_to_repr
from discopt.dae import (
    ContinuousSet,
    DAEBuilder,
    FDBuilder,
    MOLBuilder,
    SpatialSet,
)

# ─────────────────────────────────────────────────────────────
# Model factories: one per DAE builder
# ─────────────────────────────────────────────────────────────


def _issue_repro():
    """The model from the issue, verbatim."""
    m = dm.Model("dae_ctrl")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=4, ncp=2)
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=1.0, bounds=(-2, 2))
    dae.add_control("u", bounds=(-2, 2))
    dae.set_ode(lambda t, s, a, c: {"x": -(s["x"] ** 2) + c["u"]})
    dae.discretize()
    xv = dae.get_state("x")
    m.minimize(
        dae.integral(lambda t, s, a, c: c["u"] ** 2)
        + 10.0 * (xv[xv.shape[0] - 1, xv.shape[1] - 1] - 0.2) ** 2
    )
    return m


def _collocation_vector_control():
    """``n_components > 1`` takes the ``var[:, c, None]`` branch."""
    m = dm.Model("dae_vec_ctrl")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=3, ncp=3, scheme="legendre")
    dae = DAEBuilder(m, cs)
    dae.add_state("x", initial=0.5, bounds=(-3, 3))
    dae.add_control("u", n_components=2, bounds=(-1, 1))
    dae.set_ode(lambda t, s, a, c: {"x": c["u"][0] - 0.5 * c["u"][1] * s["x"]})
    dae.discretize()
    m.minimize(dae.integral(lambda t, s, a, c: s["x"] ** 2 + c["u"][0] ** 2 + c["u"][1] ** 2))
    return m


def _finite_difference():
    m = dm.Model("fd_ctrl")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=6)
    fd = FDBuilder(m, cs, method="backward")
    fd.add_state("x", initial=1.0, bounds=(-2, 2))
    fd.add_control("u", bounds=(-2, 2))
    fd.set_ode(lambda t, s, a, c: {"x": -(s["x"] ** 2) + c["u"]})
    fd.discretize()
    x = fd.get_state("x")
    u = fd._vars["u"]
    m.minimize(dm.sum(u**2) + 10.0 * (x[x.shape[0] - 1] - 0.2) ** 2)
    return m


def _mol(time_method):
    m = dm.Model(f"mol_{time_method}")
    ts = ContinuousSet("t", bounds=(0, 1), nfe=3, ncp=2)
    ss = SpatialSet("z", bounds=(0, 1), npts=3)
    kw = {"fd_method": "backward"} if time_method == "finite_difference" else {}
    mol = MOLBuilder(m, ts, ss, time_method=time_method, **kw)
    mol.add_field("w", initial=0.0, bounds=(-5, 5))
    mol.add_control("q", bounds=(0, 2))
    mol.set_pde(lambda t, z, f, fz, fzz, c: {"w": fzz["w"] + c["q"]})
    mol.discretize()
    w = mol.get_field("w")
    m.minimize(dm.sum((w - 0.1) ** 2))
    return m


def _neural_collocation():
    """A trainable surrogate inside collocation: ``TrainableNetwork`` squeezes
    its output with ``h[..., 0]`` where ``h`` is a matmul whose shape the
    modelling layer does not track -- the ``Ellipsis`` half of #1516."""
    from discopt.ml import TrainableNetwork

    m = dm.Model("neural_dae")
    net = TrainableNetwork(m, [1, 3, 1], activation="tanh", weight_bounds=(-3, 3), name="r")
    cs = ContinuousSet("t", bounds=(0, 1), nfe=3, ncp=2)
    dae = DAEBuilder(m, cs)
    dae.add_state("c", initial=1.0, bounds=(0, 2))
    dae.add_control("u", bounds=(0, 1))
    dae.set_ode(lambda t, s, a, c: {"c": -net(s["c"]) + c["u"]})
    dae.discretize()
    cv = dae.get_state("c")
    m.minimize((cv[cv.shape[0] - 1, cv.shape[1] - 1] - 0.5) ** 2)
    return m


BUILDERS = {
    "issue_repro": _issue_repro,
    "neural_collocation": _neural_collocation,
    "collocation_vector_control": _collocation_vector_control,
    "finite_difference": _finite_difference,
    "mol_finite_difference": lambda: _mol("finite_difference"),
    "mol_collocation": lambda: _mol("collocation"),
}


def _has_newaxis_index(model, marker=None) -> int:
    """How many subscripts in the model use ``marker`` (``None`` or ``...``)."""
    from discopt.modeling.core import Expression, IndexExpression

    seen: set[int] = set()
    hits = 0
    stack = [model._objective.expression] + [c.body for c in model._constraints]
    while stack:
        e = stack.pop()
        if id(e) in seen or not isinstance(e, Expression):
            continue
        seen.add(id(e))
        if isinstance(e, IndexExpression):
            idx = e.index if isinstance(e.index, tuple) else (e.index,)
            hits += sum(1 for i in idx if i is marker)
        for attr in ("base", "left", "right", "operand"):
            child = getattr(e, attr, None)
            if child is not None:
                stack.append(child)
        for attr in ("args", "terms"):
            stack.extend(getattr(e, attr, None) or ())
    assert seen, "walk visited no nodes"
    return hits


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_dae_model_converts(name):
    m = BUILDERS[name]()
    repr_ = model_to_repr(m, getattr(m, "_builder", None))
    assert repr_.n_constraints == len(m._constraints)
    assert repr_.n_vars == sum(v.size for v in m._variables)


def test_issue_repro_exercises_newaxis():
    """Pin that the models still carry the constructs the fix is about, so the
    conversion test above cannot pass vacuously if the builders stop using them."""
    assert _has_newaxis_index(_issue_repro()) >= 1
    assert _has_newaxis_index(_collocation_vector_control()) >= 1
    assert _has_newaxis_index(_neural_collocation(), Ellipsis) >= 1


# ─────────────────────────────────────────────────────────────
# NewAxis semantics: shape and element selection must equal numpy's
# ─────────────────────────────────────────────────────────────

_OP_CONST, _OP_VAR, _OP_ADD, _OP_SUB, _OP_MUL, _OP_DIV, _OP_POW, _OP_NEG = 1, 2, 3, 4, 5, 6, 7, 8
_OP_SUMOVER = 10


def _run_program(prog, x):
    """Interpret ``tape_program_expanded`` output (the ops these tests emit)."""
    op, a, b, k, args_flat, args_ptr, obj_root, row_roots, rows_per = prog
    v = np.zeros(len(op))
    for i in range(len(op)):
        o = int(op[i])
        if o == _OP_CONST:
            v[i] = k[i]
        elif o == _OP_VAR:
            v[i] = x[int(k[i])]
        elif o == _OP_ADD:
            v[i] = v[a[i]] + v[b[i]]
        elif o == _OP_SUB:
            v[i] = v[a[i]] - v[b[i]]
        elif o == _OP_MUL:
            v[i] = v[a[i]] * v[b[i]]
        elif o == _OP_DIV:
            v[i] = v[a[i]] / v[b[i]]
        elif o == _OP_POW:
            v[i] = v[a[i]] ** v[b[i]]
        elif o == _OP_NEG:
            v[i] = -v[a[i]]
        elif o == _OP_SUMOVER:
            v[i] = sum(v[j] for j in args_flat[args_ptr[i] : args_ptr[i + 1]])
        else:
            raise AssertionError(f"opcode {o} not handled by the test interpreter")
    return v[obj_root], np.array([v[r] for r in row_roots]), list(rows_per)


@pytest.mark.parametrize(
    "build, expected",
    [
        # (description-free) lambda(x2, y3) -> expression, numpy reference
        (lambda x, y: x[:, None] * y, lambda X, Y: X[:, None] * Y),
        (lambda x, y: y * x[:, None] - 1.0, lambda X, Y: Y * X[:, None] - 1.0),
        (lambda x, y: y[:, 1, None] + x[:, None], lambda X, Y: Y[:, 1, None] + X[:, None]),
        (lambda x, y: y[None, :, :] * 2.0, lambda X, Y: Y[None, :, :] * 2.0),
        (lambda x, y: y[:, None] - y[:, None, 0:1], lambda X, Y: Y[:, None] - Y[:, None, 0:1]),
        (lambda x, y: y[..., -1] * x, lambda X, Y: Y[..., -1] * X),
        (lambda x, y: y[None, -1] * x[None, :, None], lambda X, Y: Y[None, -1] * X[None, :, None]),
        # `...` on a matmul, whose shape the modelling layer leaves unknown: the
        # ellipsis must be resolved against the Rust-side shape.
        (
            lambda x, y: (y @ np.array([[1.0], [-2.0]]))[..., 0] + x,
            lambda X, Y: (Y @ np.array([[1.0], [-2.0]]))[..., 0] + X,
        ),
        (
            lambda x, y: (y @ np.array([[1.0], [-2.0]]))[..., None] * 3.0,
            lambda X, Y: (Y @ np.array([[1.0], [-2.0]]))[..., None] * 3.0,
        ),
    ],
)
def test_newaxis_rows_match_numpy(build, expected):
    """The expanded Rust program must produce numpy's rows, in numpy's order.

    ``x`` has shape (3,) and ``y`` (3, 2), so a mis-placed or mis-counted length-1
    axis either fails to broadcast or pairs the wrong elements -- both of which a
    value comparison at a random point catches."""
    m = dm.Model("newaxis")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    y = m.continuous("y", shape=(3, 2), lb=-2, ub=2)
    expr = build(x, y)
    m.subject_to(expr <= 100.0)
    m.minimize(dm.sum(x))
    repr_ = model_to_repr(m)
    prog = repr_.tape_program_expanded()

    rng = np.random.default_rng(1516)
    n_compared = 0
    for _ in range(5):
        X = rng.uniform(-2, 2, 3)
        Y = rng.uniform(-2, 2, (3, 2))
        _, rows, rows_per = _run_program(prog, np.concatenate([X, Y.ravel()]))
        ref = np.asarray(expected(X, Y), dtype=float).ravel() - 100.0
        assert rows_per == [ref.size]
        np.testing.assert_allclose(rows, ref, rtol=1e-12, atol=1e-12)
        n_compared += ref.size
    assert n_compared > 0


def test_unsupported_index_is_refused_by_name():
    """Advanced (array) indexing has no IndexSpec form: refused with a message
    naming it, not pyo3's bare integer-conversion error."""
    from discopt.modeling.core import IndexExpression

    m = dm.Model("adv")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    m.minimize(dm.sum(IndexExpression(x, (np.array([0, 2]),))))
    with pytest.raises(TypeError, match="unsupported index component"):
        model_to_repr(m)


# ─────────────────────────────────────────────────────────────
# The solver now actually uses the repr on DAE models
# ─────────────────────────────────────────────────────────────


def test_solver_runs_rust_presolve_on_dae(monkeypatch, caplog):
    """Root presolve ran on the repr, and no repr-unavailable warning fired.

    Before the fix ``_model_repr`` was ``None`` for this model, so
    ``run_root_presolve`` was never called (count 0)."""
    import discopt._relax.presolve_pipeline as pp
    import discopt.solver as solver_mod

    calls = []
    orig = pp.run_root_presolve

    def spy(repr_, *a, **k):
        assert repr_ is not None
        out = orig(repr_, *a, **k)
        calls.append(out[1])
        return out

    monkeypatch.setattr(pp, "run_root_presolve", spy)
    m = _issue_repro()
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        res = m.solve(time_limit=20, max_nodes=50)
    assert len(calls) == 1, "root presolve never ran on the DAE model"
    assert isinstance(calls[0], dict) and calls[0], "presolve returned no stats"
    assert not [r for r in caplog.records if "Rust model repr unavailable" in r.getMessage()]
    assert res.objective is not None
    assert solver_mod._in_tree_presolve_global_calls() >= 0  # telemetry readable


def test_fbbt_box_contains_dae_solution():
    """Soundness of the newly-running FBBT on a DAE model: the box it derives
    must contain a feasible point (the solver's incumbent, re-verified)."""
    m = _issue_repro()
    res = m.solve(time_limit=20, max_nodes=50)
    assert res.x is not None

    # ``node_limit``: the native spatial kernel's name for a max_nodes stop holding a
    # verified incumbent (accepted since #1153); it serves this model since
    # DISCOPT_NATIVE_KERNEL_VECTOR_VARS. The incumbent is what the check below needs.
    assert str(res.status) in (
        "SolveStatus.OPTIMAL",
        "SolveStatus.FEASIBLE",
        "optimal",
        "feasible",
        "node_limit",
    )
    repr_ = model_to_repr(m)
    lbs, ubs = repr_.fbbt(max_iter=50, tol=1e-9)
    lbs, ubs = np.asarray(lbs), np.asarray(ubs)
    # `fbbt` returns one hull per variable BLOCK; every element must lie in it.
    assert lbs.shape == (len(m._variables),)
    n_checked = 0
    for j, v in enumerate(m._variables):
        vals = np.asarray(res.x[v.name], dtype=float).ravel()
        assert np.all(vals >= lbs[j] - 1e-6), f"FBBT lb cut off {v.name}"
        assert np.all(vals <= ubs[j] + 1e-6), f"FBBT ub cut off {v.name}"
        n_checked += vals.size
    assert n_checked == sum(v.size for v in m._variables) > 0


def test_repr_failure_is_a_warning(monkeypatch, caplog):
    """A model the repr builder refuses still solves, but says so loudly."""
    import discopt._rust as R

    def boom(*a, **k):
        raise TypeError("synthetic repr failure (#1516)")

    monkeypatch.setattr(R, "model_to_repr", boom)
    m = dm.Model("warn")
    x = m.continuous("x", lb=0, ub=2)
    m.minimize((x - 1) ** 2)
    with caplog.at_level(logging.WARNING, logger="discopt.solver"):
        m.solve(time_limit=10)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Rust model repr unavailable" in s and "synthetic repr failure" in s for s in msgs)
