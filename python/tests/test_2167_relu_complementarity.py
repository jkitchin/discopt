"""#2167 slice 1: the ``relu_complementarity`` NN strategy.

Each unstable ReLU neuron becomes ``y - s == z``, ``0 <= y ⊥ s >= 0`` as a
declared :class:`~discopt.mpec.Complementarity` relation (role
``FROM_ACTIVATION``), with no binary and no big-M row. The exact lowerings
(``solve_mpec(method="sos1" | "gdp")``) must reproduce ``relu_bigm``'s certified
objective on the same network, and the relations must keep their provenance
through rebuilding passes and serialization.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
from discopt.ml.formulations import NNFormulation  # noqa: E402
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition  # noqa: E402
from discopt.ml.scaling import OffsetScaling  # noqa: E402
from discopt.modeling.core import Model, VarType  # noqa: E402
from discopt.mpec import ComplementarityRole, solve_mpec  # noqa: E402
from discopt.mpec_report import relation_residuals  # noqa: E402

pytestmark = pytest.mark.requires_pounce


def _relu_net(seed=42, n_in=2, hidden=(3,), with_bounds=True):
    """ReLU hidden layers -> LINEAR scalar output."""
    rng = np.random.RandomState(seed)
    layers = []
    width = n_in
    for h in hidden:
        layers.append(DenseLayer(rng.randn(width, h), rng.randn(h), Activation.RELU))
        width = h
    layers.append(DenseLayer(rng.randn(width, 1), rng.randn(1), Activation.LINEAR))
    bounds = (-np.ones(n_in), np.ones(n_in)) if with_bounds else None
    return NetworkDefinition(layers, input_bounds=bounds)


def _embed(net, strategy, sense="min", scaling=None):
    m = Model(f"nn_{strategy}")
    nn = NNFormulation(m, net, strategy=strategy, scaling=scaling)
    nn.formulate()
    obj = dm.sum(nn.outputs)
    if sense == "min":
        m.minimize(obj)
    else:
        m.maximize(obj)
    return m, nn


def _point(m, res):
    """``{id(var): value}`` from a solve result, for the residual report."""
    return {id(v): np.asarray(res.x[v.name], dtype=np.float64) for v in m._variables}


# ── structure ──


def test_emits_relations_no_binaries_no_bigm():
    net = _relu_net(hidden=(4, 3))
    m, nn = _embed(net, "relu_complementarity")
    assert not any(v.var_type == VarType.BINARY for v in m._variables)
    assert not any("bigm" in (c.name or "") for c in m._constraints)

    # one relation per unstable neuron, as relu_bigm counts its indicators
    mb, _ = _embed(net, "relu_bigm")
    n_q = sum(v.var_type == VarType.BINARY for v in mb._variables)
    assert n_q > 0
    assert len(nn.relations) == n_q

    for rel in nn.relations:
        assert rel.role is ComplementarityRole.FROM_ACTIVATION
        assert rel.is_symmetric_nonnegative
        assert rel.parent.startswith("nn.layer[")
        assert rel.index is not None and len(rel.index) == 1
    # registered on the model, not lowered
    assert {id(r) for r in nn.relations} <= {id(r) for r in m._complementarities}
    assert not any(r.is_lowered_into(m) for r in nn.relations)


def test_plain_solve_refuses_unlowered_relations():
    m, _ = _embed(_relu_net(), "relu_complementarity")
    with pytest.raises(NotImplementedError, match="no lowering"):
        m.solve()


def test_relations_empty_for_other_strategies():
    _, nn = _embed(_relu_net(), "relu_bigm")
    assert nn.relations == []


def test_relations_before_formulate_raises():
    nn = NNFormulation(Model(), _relu_net(), strategy="relu_complementarity")
    with pytest.raises(RuntimeError, match="formulate"):
        nn.relations


def test_ideal_cuts_refused():
    with pytest.raises(ValueError, match="ideal_cuts"):
        NNFormulation(Model(), _relu_net(), strategy="relu_complementarity", ideal_cuts=True)


def test_unbounded_inputs_accepted_all_neurons_unstable():
    net = _relu_net(hidden=(3, 2), with_bounds=False)
    with pytest.raises(ValueError, match="input_bounds"):
        _embed(net, "relu_bigm")
    m, nn = _embed(net, "relu_complementarity")
    assert len(nn.relations) == 5
    for v in m._variables:
        lb = np.asarray(v.lb, dtype=np.float64)
        ub = np.asarray(v.ub, dtype=np.float64)
        assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), v.name
        assert not np.isnan(lb).any() and not np.isnan(ub).any(), v.name


# ── exact lowerings reproduce relu_bigm ──

_NETS = [
    pytest.param(dict(seed=42, hidden=(3,)), "min", id="1x3-min"),
    pytest.param(dict(seed=42, hidden=(3,)), "max", id="1x3-max"),
    pytest.param(dict(seed=7, n_in=3, hidden=(4, 3)), "min", id="2layer-min"),
    pytest.param(dict(seed=11, n_in=2, hidden=(5, 4)), "max", id="2layer-max"),
]


@pytest.mark.parametrize("method", ["sos1", "gdp"])
@pytest.mark.parametrize("kw,sense", _NETS)
def test_exact_lowering_matches_relu_bigm(kw, sense, method):
    net = _relu_net(**kw)
    mb, _ = _embed(net, "relu_bigm", sense)
    rb = mb.solve()
    assert rb.status == "optimal" and rb.gap_certified

    mc, nn = _embed(net, "relu_complementarity", sense)
    rc = solve_mpec(mc, nn.relations, method=method)
    assert rc.status == "optimal" and rc.gap_certified
    assert rc.objective == pytest.approx(rb.objective, abs=1e-6, rel=1e-6)

    # the incumbent's inputs reproduce its output through the true network
    x_in = np.asarray(rc.x["nn_input"], dtype=np.float64)
    assert float(net.forward(x_in)[0]) == pytest.approx(rc.objective, abs=1e-5)

    # source residuals vanish at the certified point
    assert rc.mpec_report is not None and rc.mpec_report.source_satisfied
    rr = relation_residuals(mc, nn.relations, _point(mc, rc))
    assert len(rr) == len(nn.relations)
    assert max(r.complementarity.value for r in rr) <= 1e-6


def test_exact_lowering_with_scaling_matches_relu_bigm():
    net = _relu_net(seed=3, hidden=(4,))
    sc = OffsetScaling(
        x_offset=np.array([0.5, -0.2]),
        x_factor=np.array([2.0, 0.5]),
        y_offset=np.array([1.0]),
        y_factor=np.array([3.0]),
    )
    mb, _ = _embed(net, "relu_bigm", scaling=sc)
    rb = mb.solve()
    mc, nn = _embed(net, "relu_complementarity", scaling=sc)
    rc = solve_mpec(mc, nn.relations, method="sos1")
    assert rb.gap_certified and rc.gap_certified
    assert rc.objective == pytest.approx(rb.objective, abs=1e-6, rel=1e-6)


# ── residual report ──


def test_residual_is_min_y_s_at_perturbed_point():
    net = _relu_net(hidden=(3,))
    mc, nn = _embed(net, "relu_complementarity")
    rc = solve_mpec(mc, nn.relations, method="sos1")
    point = _point(mc, rc)
    # push every unstable pair off complementarity by +delta on both sides
    delta = 0.25
    for rel in nn.relations:
        s_var = rel.g
        point[id(s_var)] = point[id(s_var)] + delta
    by_name = {v.name: v for v in mc._variables}
    y_var = by_name["nn_z_0"]
    y = point[id(y_var)].copy()
    for rel in nn.relations:
        y[rel.index[0]] += delta
    point[id(y_var)] = y

    rr = relation_residuals(mc, nn.relations, point)
    assert len(rr) == len(nn.relations)  # the probe measured every pair
    checked = 0
    for r in rr:
        expect = min(r.f_value, r.g_value)
        assert r.complementarity.value == pytest.approx(expect, abs=1e-12)
        assert r.complementarity.value >= delta - 1e-12
        checked += 1
    assert checked == len(nn.relations) > 0


# ── provenance through rebuilding passes and serialization ──


def _relation_signature(model):
    return sorted(
        (r.name, r.role.value, r.parent, tuple(r.index or ())) for r in model._complementarities
    )


def test_provenance_survives_deepcopy_and_clone():
    m, nn = _embed(_relu_net(hidden=(3, 2)), "relu_complementarity")
    sig = _relation_signature(m)
    assert len(sig) == len(nn.relations) > 0
    assert _relation_signature(copy.deepcopy(m)) == sig
    assert _relation_signature(m.clone()) == sig


def test_provenance_survives_serialization():
    from discopt.serialize import dumps, loads

    m, nn = _embed(_relu_net(hidden=(3,)), "relu_complementarity")
    m2 = loads(dumps(m))
    assert _relation_signature(m2) == _relation_signature(m)
    assert all(r.role is ComplementarityRole.FROM_ACTIVATION for r in m2._complementarities)


def test_provenance_survives_gdp_lowering():
    net = _relu_net(hidden=(3,))
    m, nn = _embed(net, "relu_complementarity")
    sig = _relation_signature(m)
    res = solve_mpec(m, nn.relations, method="gdp")
    assert res.gap_certified
    assert _relation_signature(m) == sig
    assert all(r.lowering_in(m) == "gdp" for r in nn.relations)
    roles = {r.role for r in res.mpec_report.relations}
    assert roles == {ComplementarityRole.FROM_ACTIVATION.value}


def test_exact_lowering_refuses_unbounded_operands():
    """No input box: the strategy declares the pairs, but today's big-M lowering
    cannot bound them and must refuse rather than invent a big-M."""
    m, nn = _embed(_relu_net(hidden=(3,), with_bounds=False), "relu_complementarity")
    with pytest.raises(ValueError, match="infinite"):
        solve_mpec(m, nn.relations, method="sos1")


# ── local mode ──


@pytest.mark.parametrize("kw,sense", _NETS)
def test_scholtes_is_local_and_near_certified_optimum(kw, sense):
    net = _relu_net(**kw)
    mb, _ = _embed(net, "relu_bigm", sense)
    rb = mb.solve()
    mc, nn = _embed(net, "relu_complementarity", sense)
    rc = solve_mpec(mc, nn.relations, method="scholtes")
    assert rc.status.startswith("local_")
    assert not rc.gap_certified
    if rc.status == "local_optimal":
        assert rc.objective == pytest.approx(rb.objective, abs=1e-4)
        rr = relation_residuals(mc, nn.relations, _point(mc, rc))
        assert len(rr) == len(nn.relations)
        assert max(r.complementarity.value for r in rr) <= 1e-3


# ── slice 2: multistart local mode ──


def test_start_point_is_exact_forward_pass():
    sc = OffsetScaling(
        x_offset=np.array([0.5, -0.2]),
        x_factor=np.array([2.0, 0.5]),
        y_offset=np.array([1.0]),
        y_factor=np.array([3.0]),
    )
    net = _relu_net(seed=3, hidden=(4, 3))
    m, nn = _embed(net, "relu_complementarity", scaling=sc)
    starts = nn.local_starts(5, seed=1)
    assert len(starts) == 5
    from discopt.mpec_report import source_residual_report
    from discopt.validation.feasibility import verify_point
    from discopt.warm_start import validate_initial_solution

    checked = 0
    for st in starts:
        x = validate_initial_solution(m, st)
        # every NN variable is set, so the start is exactly on the network graph
        assert verify_point(m, x).ok
        assert source_residual_report(m, x_flat=x).source_satisfied
        x_in = np.asarray(st[nn.inputs])
        assert np.all(np.abs(x_in) <= 1.0)
        x_scaled = (x_in - sc.x_offset) / sc.x_factor
        assert float(st[nn.outputs][0]) == pytest.approx(
            float(net.forward(x_scaled)[0]) * 3.0 + 1.0, abs=1e-12
        )
        checked += 1
    assert checked == 5


def test_local_starts_needs_complementarity_strategy():
    _, nn = _embed(_relu_net(), "relu_bigm")
    with pytest.raises(RuntimeError, match="relu_complementarity"):
        nn.local_starts(2)


@pytest.mark.parametrize("kw,sense", _NETS)
def test_multistart_is_local_and_verified(kw, sense):
    from discopt.mpec_report import accept_local_incumbent, source_residual_report

    net = _relu_net(**kw)
    mb, _ = _embed(net, "relu_bigm", sense)
    rb = mb.solve()
    m, nn = _embed(net, "relu_complementarity", sense)
    res = nn.solve_local(n_starts=4, seed=0)
    assert res.status in ("local_optimal", "local_infeasible")
    assert not res.gap_certified and res.bound is None
    assert len(res.multistart) == 4
    assert all(
        r.scholtes_status.startswith("local_") or r.scholtes_status == "error"
        for r in res.multistart
    )
    assert res.status == "local_optimal"
    from discopt.mpec_report import _flat_from_result

    x = _flat_from_result(m, res)
    assert accept_local_incumbent(m, None, x_flat=x) == pytest.approx(res.objective)
    rep = source_residual_report(m, x_flat=x)
    assert rep.source_satisfied
    assert rep.complementarity.value <= 1e-9  # polish makes it exact, not sqrt(t)
    # never better than the certified optimum (a local point is a primal point)
    if sense == "min":
        assert res.objective >= rb.objective - 1e-6
    else:
        assert res.objective <= rb.objective + 1e-6
    # the reported objective is the network's true output at the reported input
    x_in = np.asarray(res.x["nn_input"], dtype=np.float64)
    assert float(net.forward(x_in)[0]) == pytest.approx(res.objective, abs=1e-6)
    best = [r.verified_objective for r in res.multistart if r.verified_objective is not None]
    assert res.objective == pytest.approx(min(best) if sense == "min" else max(best))


def test_multistart_refuses_unverifiable_point(monkeypatch):
    """With the verifier refusing every point, nothing becomes ``x``."""
    import discopt.mpec_report as rep

    m, nn = _embed(_relu_net(), "relu_complementarity")
    calls = []

    def refuse(model, result, *, x_flat=None):
        calls.append(1)
        return None

    monkeypatch.setattr(rep, "accept_local_incumbent", refuse)
    res = nn.solve_local(n_starts=2, seed=0)
    assert calls, "the verifier was never consulted"
    assert res.status == "local_infeasible"
    assert res.x is None and res.objective is None and not res.gap_certified
    assert all(r.verified_objective is None for r in res.multistart)


def test_multistart_without_polish_stays_local():
    """``polish=False`` runs no polish NLP; whatever is reported is still a point
    ``accept_local_incumbent`` vouched for (an unpolished Scholtes point passes the
    1e-6 source check only when it happens to be that close to complementary --
    the H1 entry rows measured 1e-7 to 3e-6, so neither outcome is pinned here)."""
    m, nn = _embed(_relu_net(seed=7, n_in=3, hidden=(4, 3)), "relu_complementarity")
    res = nn.solve_local(n_starts=3, seed=0, polish=False)
    assert len(res.multistart) == 3
    assert all(r.polish_status is None for r in res.multistart)
    assert not res.gap_certified and res.bound is None
    verified = [r.verified_objective for r in res.multistart if r.verified_objective is not None]
    assert (res.status == "local_optimal") == bool(verified)
    if verified:
        assert res.objective == pytest.approx(min(verified))


def test_multistart_rejects_solve_kwargs_and_empty_starts():
    from discopt.mpec import solve_mpec_multistart

    m, nn = _embed(_relu_net(), "relu_complementarity")
    with pytest.raises(ValueError, match="at least one start"):
        solve_mpec_multistart(m, nn.relations, [])
    with pytest.raises(ValueError, match="entries"):
        solve_mpec_multistart(m, nn.relations, [np.zeros(1)])


def test_multistart_skips_polish_for_expression_operands():
    """A pair over a general expression cannot be fixed by a bound; the driver
    records the skip rather than pretending to polish."""
    from discopt.mpec import complementarity, solve_mpec_multistart

    m = Model("expr_pair")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    res = solve_mpec_multistart(m, [complementarity(x + 0.0 * y, y)], [None])
    assert res.multistart[0].polish_status is None
    assert "polish skipped" in res.multistart[0].note
    assert not res.gap_certified
