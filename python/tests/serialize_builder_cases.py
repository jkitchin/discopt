"""One small model per model-producing builder, for the WP-6 serialization sweep.

Plan: ``docs/dev/serialization-completion-plan.md`` WP-6 (D12). Shared by
``scripts/serialize_builder_sweep.py`` (which classifies every case and prints the
table) and ``test_serialize_builders.py`` (which pins each outcome). Each case is a
zero-argument function returning a fresh ``Model``; the snippets are adapted from
the builders' own tests, cited per case.

* :data:`ROUND_TRIPS` -- cases that must round-trip through ``dumps``/``loads``.
* :data:`REFUSED` -- cases ``dumps`` must refuse with a ``SerializationError``
  whose message matches, paired with the remedy the message names, which the
  tests run: ``"pickle"`` (same-version transport keeps the model), or a
  ``Remedy(source, apply)`` (save the source model, reload it, re-apply the pass;
  the result must be fingerprint-equal to applying the pass without the trip).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
from discopt.modeling.core import Constant


@dataclass(frozen=True)
class Remedy:
    """Save ``source()``, reload it, and ``apply(reloaded)``."""

    source: Callable[[], Any]
    apply: Callable[[Any], Any]


@dataclass(frozen=True)
class Refusal:
    build: Callable[[], Any]
    match: str
    remedy: Any  # "pickle" | Remedy | Callable[[], Any] (an attached rebuild)


# ── DAE (test_dae.py, test_dae_fd_mol.py, test_dae_fit.py) ──────────────────


def dae_collocation():
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("colloc")
    dae = DAEBuilder(m, ContinuousSet("t", bounds=(0, 2), nfe=3, ncp=2))
    dae.add_state("x", initial=1.0, bounds=(-5, 5))
    dae.add_algebraic("y", bounds=(-10, 10))
    dae.set_ode(lambda t, s, a, c: {"x": -s["x"]})
    dae.set_algebraic(lambda t, s, a, c: {"y": a["y"] - 2 * s["x"]})
    dae.discretize()
    m.minimize(dae.get_state("x")[-1, -1] ** 2)
    return m


def dae_collocation_control():
    from discopt.dae import ContinuousSet, DAEBuilder

    m = dm.Model("colloc_ctrl")
    dae = DAEBuilder(m, ContinuousSet("t", bounds=(0, 2), nfe=3, ncp=2))
    dae.add_state("x", initial=1.0, bounds=(-5, 5))
    dae.add_control("u", bounds=(-1, 1))
    dae.set_ode(lambda t, s, a, c: {"x": -s["x"] + c["u"]})
    dae.discretize()
    m.minimize(dae.get_state("x")[-1, -1] ** 2)
    return m


def dae_finite_difference():
    from discopt.dae import ContinuousSet, FDBuilder

    m = dm.Model("fd")
    fd = FDBuilder(m, ContinuousSet("t", bounds=(0, 1), nfe=5), method="backward")
    fd.add_state("x", initial=0.0, bounds=(-10, 10))
    fd.add_control("u", bounds=(-1, 1))
    fd.set_ode(lambda t, s, a, c: {"x": c["u"]})
    fd.discretize()
    m.minimize((fd.get_state("x")[-1] - 0.5) ** 2)
    return m


def dae_method_of_lines():
    from discopt.dae import ContinuousSet, MOLBuilder, SpatialSet

    m = dm.Model("mol")
    mol = MOLBuilder(
        m,
        ContinuousSet("t", bounds=(0, 1), nfe=5),
        SpatialSet("z", bounds=(0, 1), npts=3),
        time_method="finite_difference",
        fd_method="backward",
    )
    mol.add_field("u", initial=0.0, bounds=(-5, 5))
    mol.add_control("q", bounds=(0, 10))
    mol.set_pde(lambda t, z, f, fz, fzz, c: {"u": fzz["u"] + c["q"]})
    mol.discretize()
    m.minimize(0)
    return m


def dae_fit_trajectories():
    from discopt.dae import Trajectory, fit_trajectories
    from discopt.ml.trainable import TrainableKernelExpansion

    m = dm.Model("fit")
    ke = TrainableKernelExpansion(m, np.linspace(0, 1.05, 4), lengthscale=0.2, name="k")
    t = np.linspace(0, 1, 5)
    tr = Trajectory(t_data=t, y_data={"cA": np.exp(-t)}, initial={"cA": 1.0})
    fit = fit_trajectories(
        m,
        trajectories=[tr],
        states=[("cA", dict(bounds=(0.0, 1.5)))],
        rhs=lambda t, s, a, u: {"cA": -ke(s["cA"])},
        t_span=(0.0, 1.0),
        nfe=3,
        ncp=2,
    )
    m.minimize(fit.least_squares() + 1e-4 * ke.l2_penalty())
    return m


# ── ML (test_nn_predictor.py, test_tree.py, test_linear_tree.py, test_nn_trainable.py) ──


def _net(activation):
    from discopt.ml.network import Activation, DenseLayer, NetworkDefinition

    r = np.random.RandomState(0)
    return NetworkDefinition(
        [
            DenseLayer(r.randn(2, 3), r.randn(3), activation),
            DenseLayer(r.randn(3, 1), r.randn(1), Activation.LINEAR),
        ]
    )


def _predictor(method, activation_name):
    def build():
        from discopt.ml.network import Activation
        from discopt.ml.predictor import add_predictor

        m = dm.Model(method)
        x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
        out, _ = add_predictor(
            m, x, _net(getattr(Activation, activation_name)), method=method, prefix="p"
        )
        m.minimize(out[0])
        return m

    build.__name__ = f"ml_{method}_{activation_name.lower()}"
    return build


def ml_tree_ensemble():
    from discopt.ml.predictor import add_predictor
    from discopt.ml.tree import DecisionTree, TreeEnsembleDefinition

    tree = DecisionTree(
        n_features=2,
        feature=np.array([0, -1, -1]),
        threshold=np.array([0.5, 0.0, 0.0]),
        left_child=np.array([1, -1, -1]),
        right_child=np.array([2, -1, -1]),
        value=np.array([0.0, 1.0, 2.0]),
    )
    ens = TreeEnsembleDefinition(trees=[tree], n_features=2, input_bounds=(np.zeros(2), np.ones(2)))
    m = dm.Model("tree")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    out, _ = add_predictor(m, x, ens, prefix="tf")
    m.minimize(out[0])
    return m


def _linear_tree(encoding):
    def build():
        from discopt.ml import LinearTreeDefinition
        from discopt.ml.predictor import add_predictor
        from test_linear_tree import LB, UB, _spec

        m = dm.Model("ltree")
        x = m.continuous("x", shape=(2,), lb=LB, ub=UB)
        definition = LinearTreeDefinition.from_spec(_spec(seed=2))
        out, _ = add_predictor(m, x, definition, method=encoding, prefix="lt")
        m.maximize(out[0])
        return m

    build.__name__ = f"ml_linear_tree_{encoding}"
    return build


def ml_trainable_network():
    from discopt.ml.trainable import TrainableNetwork

    m = dm.Model("trainable")
    net = TrainableNetwork(m, [1, 4, 1], activation="tanh", name="n1")
    s = m.continuous("s", lb=0.0, ub=1.0)
    m.minimize((net(s) - 0.5) ** 2 + net.l2_penalty())
    return m


def ml_trainable_l2_only():
    from discopt.ml.trainable import TrainableNetwork

    m = dm.Model("trainable_l2")
    net = TrainableNetwork(m, [1, 4, 1], activation="tanh", name="n1")
    m.minimize(net.l2_penalty())
    return m


def ml_frozen_network():
    from discopt.ml.predictor import add_predictor
    from discopt.ml.trainable import TrainableNetwork
    from discopt.warm_start import unflatten_solution, validate_initial_solution

    m0 = dm.Model("train")
    net = TrainableNetwork(m0, [1, 5, 3, 1], activation="softplus", name="net")
    weights = unflatten_solution(m0, validate_initial_solution(m0, net.initial_values(seed=3)))
    frozen = net.freeze(weights, input_bounds=(np.zeros(1), np.ones(1)))
    m = dm.Model("frozen")
    x = m.continuous("x", shape=(1,), lb=0, ub=1)
    out, _ = add_predictor(m, x, frozen, method="full_space", prefix="fz")
    m.minimize(out[0])
    return m


# ── GDP (test_1430_unattached_gdp_blocks.py, test_serialize_logical.py, test_1775_g1_3_state.py) ──


# ── ML readers (test_nn_readers.py): optional dependencies, see READERS ─────


def _reader_data():
    X = np.random.RandomState(0).rand(40, 2)
    return X, X @ np.array([1.0, 2.0])


def _reader_model(name, predictor):
    from discopt.ml.predictor import add_predictor

    m = dm.Model(name)
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    out, _ = add_predictor(m, x, predictor, prefix="p", input_bounds=(np.zeros(2), np.ones(2)))
    m.minimize(out[0])
    return m


def _torch_sequential():
    import torch

    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))


def ml_reader_sklearn_mlp():
    from sklearn.neural_network import MLPRegressor

    # By keyword: newer scikit-learn takes `loss` as the first positional parameter.
    mlp = MLPRegressor(hidden_layer_sizes=(3,), max_iter=50, random_state=0)
    return _reader_model("sk_mlp", mlp.fit(*_reader_data()))


def ml_reader_sklearn_gbr():
    from sklearn.ensemble import GradientBoostingRegressor

    gbr = GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=0)
    return _reader_model("sk_gbr", gbr.fit(*_reader_data()))


def ml_reader_torch_sequential():
    return _reader_model("torch_seq", _torch_sequential())


def ml_reader_onnx():
    import os
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.onnx")
        torch.onnx.export(_torch_sequential(), torch.zeros(1, 2), path, dynamo=False)
        return _reader_model("onnx", path)


def _xy(name):
    m = dm.Model(name)
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    return m, x, y


def _either_or(semantics):
    def build():
        m, x, y = _xy("either_or")
        m.either_or([[x >= 4, y >= 1], [y >= 6, x >= 1]], name="d", semantics=semantics)
        return m

    build.__name__ = f"gdp_either_or_{semantics}"
    return build


def gdp_disjunct_blocks():
    m, x, y = _xy("blocks_flat")
    a = m.make_disjunct("a")
    a.subject_to([x >= 4, y >= 1])
    b = m.make_disjunct("b")
    b.subject_to([y >= 6, x >= 1])
    m.add_disjunction([a, b])
    return m


def gdp_logical():
    from discopt.modeling.core import atleast, exactly, xor

    m = dm.Model("logical")
    Y = m.boolean("Y", shape=(4,))
    z = m.boolean("z")
    x = m.continuous("x", lb=0, ub=1)
    m.logical((Y[0] & ~Y[1]) | (Y[2] & z), name="L")
    m.logical(Y[0].implies(Y[1] | ~Y[3]))
    m.logical(atleast(2, [Y[0], Y[1], Y[2], z]))
    m.logical(xor(Y[0], Y[1]) & exactly(1, [Y[2], Y[3]]))
    m.minimize(x)
    return m


def gdp_nested_disjunction():
    m = dm.Model("nested")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.maximize(x)
    inner = m.disjunction([[x >= 1, x <= 3], [x >= 5, x <= 7]], name="inner")
    m.either_or([[inner], [x >= 15, x <= 16]], name="outer")
    return m


def gdp_nested_blocks():
    m = dm.Model("blocks")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a2 = m.make_disjunct("A2")
    a2.subject_to([y >= 8, x >= 2])
    a.add_disjunction([a1, a2])
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b])
    m.minimize(x + 3 * y)
    return m


def gdp_or_disjunct_blocks():
    m, x, y = _xy("blocks_or")
    a = m.make_disjunct("a")
    a.subject_to([x >= 4, y >= 1])
    b = m.make_disjunct("b")
    b.subject_to([y >= 6, x >= 1])
    m.add_disjunction([a, b], semantics="or")
    return m


def gdp_disjunct_local_variable():
    m, x, y = _xy("local")
    a = m.make_disjunct("a")
    w = a.variable("w", lb=0, ub=5)
    a.subject_to([x >= 4, w <= x])
    b = m.make_disjunct("b")
    b.subject_to(y >= 6)
    m.add_disjunction([a, b])
    return m


def gdp_unattached_disjunction():
    m, x, y = _xy("unattached")
    m.disjunction([[x >= 4, y >= 1], [y >= 6, x >= 1]], name="mode")
    return m


def gdp_unattached_disjunct():
    m, x, y = _xy("orphan")
    m.make_disjunct("a").subject_to(x >= 4)
    return m


# ── stochastic (test_stochastic_phase0.py) ─────────────────────────────────


def _newsvendor(risk_name):
    def build():
        from discopt import stochastic as st

        risk = {
            "expectation": st.Expectation(),
            "cvar": st.CVaR(0.8),
            "mean_cvar": st.MeanCVaR(0.5, 0.8),
        }[risk_name]
        m = dm.Model("newsvendor")
        q = m.continuous("q", lb=0, ub=100)

        def recourse(model, data, s):
            sales = model.continuous(f"sales_{s}", lb=0, ub=100)
            model.subject_to(sales <= q)
            model.subject_to(sales <= Constant(float(data["demand"])))
            return Constant(-5.0) * sales

        st.build_extensive_form(
            m,
            scenarios=st.ScenarioSet.from_list([(0.4, {"demand": 30.0}), (0.6, {"demand": 70.0})]),
            recourse_builder=recourse,
            first_stage_cost=Constant(2.0) * q,
            risk=risk,
        )
        return m

    build.__name__ = f"stochastic_{risk_name}"
    return build


def stochastic_chance_constraint():
    from discopt.stochastic import chance_constraint

    m = dm.Model("cc")
    x = m.continuous("x", lb=0, ub=100)
    chance_constraint(
        m,
        [x - Constant(d) for d in (30.0, 50.0, 70.0, 90.0)],
        [0.25] * 4,
        epsilon=0.25,
        big_m=1000.0,
    )
    m.maximize(x)
    return m


# ── robust optimization (test_robust_counterpart.py, test_1611_*, test_affine_decision_rule.py) ──


def ro_box():
    from discopt.ro import BoxUncertaintySet, RobustCounterpart

    m = dm.Model("ro_box")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    c = m.parameter("c", value=[10.0, 15.0, 8.0])
    d = m.parameter("d", value=5.0)
    m.minimize(dm.sum(c * x))
    m.subject_to(dm.sum(x) >= d, name="demand")
    RobustCounterpart(m, [BoxUncertaintySet(c, 1.0), BoxUncertaintySet(d, 0.5)]).formulate()
    return m


def ro_ellipsoidal():
    from discopt.ro import EllipsoidalUncertaintySet, RobustCounterpart

    m = dm.Model("ro_ell")
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    mu = m.parameter("mu", value=np.ones(4) * 0.1)
    m.minimize(-(mu @ x))
    m.subject_to(dm.sum(x) == 1.0)
    RobustCounterpart(m, EllipsoidalUncertaintySet(mu, rho=2.0)).formulate()
    return m


def ro_polyhedral():
    from discopt.ro import PolyhedralUncertaintySet, RobustCounterpart

    m = dm.Model("ro_poly")
    x = m.continuous("x", lb=0, ub=10)
    d = m.parameter("d", value=5.0)
    m.minimize(x)
    m.subject_to(x >= d)
    unc = PolyhedralUncertaintySet(d, A=np.array([[1.0], [-1.0]]), b=np.array([1.0, 1.0]))
    RobustCounterpart(m, unc).formulate()
    return m


def ro_budget():
    from discopt.ro import RobustCounterpart, budget_uncertainty_set

    m = dm.Model("blend")
    w = m.continuous("w", shape=(3,), lb=0, ub=1)
    s = m.parameter("s", value=np.array([5.0, 10.0, 20.0]))
    m.minimize(3 * w[0] + 2 * w[1] + w[2])
    total = w[0] + w[1] + w[2]
    m.subject_to(total >= 1, name="demand")
    m.subject_to(s @ w - 15.0 * total <= 0, name="sulfur")
    unc = budget_uncertainty_set(s, delta=np.array([1.0, 2.0, 4.0]), gamma=2.0)
    RobustCounterpart(m, unc).formulate()
    return m


def ro_affine_policy():
    from discopt.ro import AffineDecisionRule, BoxUncertaintySet, RobustCounterpart

    m = dm.Model("adr")
    x = m.continuous("x", lb=0, ub=200)
    y = m.continuous("y", lb=0, ub=100)
    d = m.parameter("d", value=80.0)
    m.minimize(2 * x + 5 * y)
    m.subject_to(x + y >= d, name="demand")
    m.subject_to(x + y <= 200, name="capacity")
    AffineDecisionRule(y, uncertain_params=d).apply()
    RobustCounterpart(m, BoxUncertaintySet(d, delta=20.0)).formulate()
    return m


# ── bilevel (test_bilevel_phase1.py, test_bilevel_phase3.py) ────────────────


def _bilevel(method, mpec_method=None):
    def build():
        from discopt.bilevel.problem import BilevelProblem

        m = dm.Model("bilevel")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x - 4 * y)
        problem = BilevelProblem(
            m,
            upper_vars=[x],
            lower_vars=[y],
            lower_objective=y,
            lower_constraints=[x + y >= 3, y <= 2 * x],
            multiplier_ub=50.0,
        )
        kwargs = {"mpec_method": mpec_method} if mpec_method else {}
        problem.formulate(method=method, **kwargs)
        return m

    build.__name__ = f"bilevel_{method}" + (f"_{mpec_method}" if mpec_method else "")
    return build


# ── geometric programming (test_gp.py) ─────────────────────────────────────


def _gp_model():
    m = dm.Model("gp")
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    m.minimize(x * y + x / y)
    m.subject_to(2.0 * x * y <= 1.0)
    return m


def gp_log_model():
    from discopt.gp import as_geometric_program

    return as_geometric_program(_gp_model()).log_model


# ── transformations (test_transformations.py) ──────────────────────────────


def _mpcc():
    from discopt import mpec

    m = dm.Model("mpcc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    return m, mpec.Complementarity(x, y)


def _piecewise_sos2():
    m = dm.Model("pw")
    x = m.continuous("x", lb=0, ub=4)
    m.piecewise(x, [0, 1, 2, 4], [0, 3, 1, 5], method="sos2", name="y")
    m.maximize(x)
    return m


def _integer_bilinear_source():
    m = dm.Model("ib")
    a = m.integer("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=3)
    m.minimize(-a * b + a)
    m.subject_to(a * b <= 7)
    return m


def _binary_multilinear_source():
    m = dm.Model("bm")
    z = m.binary("z", shape=(3,))
    m.minimize(-z[0] * z[1] * z[2] + z[0])
    m.subject_to(z[0] + z[1] >= 1)
    return m


def _nonsmooth_source():
    m = dm.Model("ns")
    x = m.continuous("x", shape=(2,), lb=-3, ub=3)
    m.minimize(dm.abs(x[0] - 0.3) + dm.maximum(x[1], -x[0]))
    m.subject_to(x[0] + x[1] == 1.0)
    return m


def _selected_second_disjunct(model):
    from discopt.gdp.tree import gdp_tree

    (disjunction,) = gdp_tree(model).disjunctions
    return [disjunction.disjuncts[1]]


def _create(name, source, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dt.create_using(name, source, **kwargs)


def _transform(name, source, make_kwargs=lambda m: {}):
    def apply(model):
        return _create(name, model, **make_kwargs(model))

    def build():
        return apply(source())

    build.__name__ = "transform_" + name.replace(".", "_")
    return build, apply


def _mpcc_transform(name, scholtes=False):
    def build():
        m, pair = _mpcc()
        kwargs = {"t": m.parameter("t", 0.01)} if scholtes else {}
        return _create(name, m, pairs=[pair], **kwargs)

    build.__name__ = "transform_" + name.replace(".", "_")
    return build


_GDP_PASSES = [
    "gdp",
    "gdp.bigm",
    "gdp.hull",
    "gdp.mbigm",
    "gdp.auto",
    "gdp.simplex",
    "gdp.binary_multiplication",
]

_TRANSFORM_ROUND_TRIPS = {
    **{f"transform:{name}": _transform(name, _either_or("select_one"))[0] for name in _GDP_PASSES},
    "transform:gdp (piecewise sos2)": _transform("gdp", _piecewise_sos2)[0],
    "transform:gdp.fix_disjuncts": _transform(
        "gdp.fix_disjuncts",
        gdp_disjunct_blocks,
        lambda m: {"selected": _selected_second_disjunct(m)},
    )[0],
    "transform:gdp.bound_pretransformation": _transform(
        "gdp.bound_pretransformation", _either_or("select_one")
    )[0],
    "transform:gdp.cuttingplane": _transform("gdp.cuttingplane", _either_or("select_one"))[0],
    "transform:mpec.gdp": _mpcc_transform("mpec.gdp"),
    "transform:mpec.sos1": _mpcc_transform("mpec.sos1"),
    "transform:mpec.scholtes": _mpcc_transform("mpec.scholtes", scholtes=True),
    "transform:core.relax_integrality": _transform(
        "core.relax_integrality", _binary_multilinear_source
    )[0],
}


def _refused_transform(name, source, make_kwargs=lambda m: {}):
    build, apply = _transform(name, source, make_kwargs)
    return Refusal(build, "Save the source model instead", Remedy(source, apply))


# ── the tables ──────────────────────────────────────────────────────────────

ROUND_TRIPS: dict[str, Callable[[], Any]] = {
    "dae:collocation": dae_collocation,
    "dae:finite_difference": dae_finite_difference,
    "dae:method_of_lines": dae_method_of_lines,
    "dae:fit_trajectories": dae_fit_trajectories,
    "dae:collocation with a control": dae_collocation_control,  # 1.6: `None` index
    "ml:full_space": _predictor("full_space", "TANH"),
    "ml:reduced_space (tanh)": _predictor("reduced_space", "TANH"),
    "ml:reduced_space (sigmoid)": _predictor("reduced_space", "SIGMOID"),
    "ml:relu_bigm": _predictor("relu_bigm", "RELU"),
    "ml:tree_ensemble": ml_tree_ensemble,
    "ml:linear_tree (hull)": _linear_tree("hull"),
    "ml:linear_tree (bigm)": _linear_tree("bigm"),
    "ml:trainable (l2 only)": ml_trainable_l2_only,
    "ml:frozen network": ml_frozen_network,
    "ml:trainable network": ml_trainable_network,  # 1.6: `Ellipsis` index
    "gdp:either_or (select_one)": _either_or("select_one"),
    "gdp:either_or (or)": _either_or("or"),
    "gdp:either_or (exactly_one_true)": _either_or("exactly_one_true"),
    "gdp:disjunct blocks": gdp_disjunct_blocks,
    "gdp:logical": gdp_logical,
    "gdp:nested disjunction()": gdp_nested_disjunction,  # 1.6: disjunction member
    "gdp:nested Disjunct blocks": gdp_nested_blocks,  # 1.7: G1.2 parent links
    "gdp:Disjunct.variable": gdp_disjunct_local_variable,  # 1.7: Disjunct records
    "gdp:OR Disjunct blocks": gdp_or_disjunct_blocks,  # 1.7: G1.4 block semantics
    "stochastic:expectation": _newsvendor("expectation"),
    "stochastic:cvar": _newsvendor("cvar"),
    "stochastic:mean_cvar": _newsvendor("mean_cvar"),
    "stochastic:chance_constraint": stochastic_chance_constraint,
    "ro:box": ro_box,
    "ro:ellipsoidal": ro_ellipsoidal,
    "ro:polyhedral": ro_polyhedral,
    "ro:budget": ro_budget,
    "ro:affine_policy": ro_affine_policy,
    "bilevel:kkt (gdp)": _bilevel("kkt", "gdp"),
    "bilevel:kkt (sos1)": _bilevel("kkt", "sos1"),
    "bilevel:strong_duality": _bilevel("strong_duality"),
    "gp:source model": _gp_model,
    "gp:log model": gp_log_model,
    **_TRANSFORM_ROUND_TRIPS,
}

#: Each refusal is final (plan §3.6): unattached blocks (#1430) and the records a
#: reformulation pass keeps of its source model. (The 1.6 and 1.7 encodings left none
#: planned.)
REFUSED: dict[str, Refusal] = {
    # #1430: not in the model; the remedy is to attach it.
    "gdp:unattached disjunction()": Refusal(
        gdp_unattached_disjunction, "never added", _either_or("select_one")
    ),
    "gdp:unattached Disjunct": Refusal(gdp_unattached_disjunct, "never added", gdp_disjunct_blocks),
    # Final (plan §3.6): a pass's records point back at its source model.
    "transform:binary.multilinear": _refused_transform(
        "binary.multilinear", _binary_multilinear_source
    ),
    "transform:integer.bilinear": _refused_transform("integer.bilinear", _integer_bilinear_source),
    "transform:integer.multilinear": _refused_transform(
        "integer.multilinear", _integer_bilinear_source
    ),
    "transform:nonsmooth.epigraph": _refused_transform("nonsmooth.epigraph", _nonsmooth_source),
    "transform:gdp.transform_current_disjunctive_state": _refused_transform(
        "gdp.transform_current_disjunctive_state",
        gdp_disjunct_blocks,
        lambda m: {"selected": _selected_second_disjunct(m)},
    ),
}

#: The ``discopt.ml.readers`` cases, which must round-trip like :data:`ROUND_TRIPS`
#: but need an optional dependency: ``key -> (modules, build)``. A case whose
#: modules are missing is skipped (pytest) or reported skipped (the sweep), never
#: counted as executed.
READERS: dict[str, tuple[tuple[str, ...], Callable[[], Any]]] = {
    "ml:reader sklearn MLPRegressor": (("sklearn",), ml_reader_sklearn_mlp),
    "ml:reader sklearn GradientBoostingRegressor": (("sklearn",), ml_reader_sklearn_gbr),
    "ml:reader torch Sequential": (("torch",), ml_reader_torch_sequential),
    "ml:reader onnx": (("torch", "onnx"), ml_reader_onnx),
}

#: Every registered transformation appears in a table: a pass added to the
#: registry without a case here fails `test_every_transformation_is_swept`.
SWEPT_TRANSFORMATIONS = {
    key.split(":", 1)[1].split(" ")[0]
    for key in (*ROUND_TRIPS, *REFUSED)
    if key.startswith("transform:")
}
