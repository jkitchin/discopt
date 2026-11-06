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
    """Continuous-q Anderson cuts for this strategy were retired by #2167's H3
    panel (median 0.0% of the gap closed at equal nodes, +25% wall)."""
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
    pytest.param(
        dict(seed=7, n_in=3, hidden=(4, 3)),
        "min",
        id="2layer-min",
        marks=[pytest.mark.slow, pytest.mark.correctness],
    ),
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


# ── slice 3: native lowering + complementarity branch ──


def _solve_native(net, sense="min", flag="0", monkeypatch=None, box=None, extra=None):
    """Embed with relu_complementarity, lower natively, solve globally."""
    if monkeypatch is not None:
        monkeypatch.setenv("DISCOPT_COMPL_BRANCH", flag)
    m, nn = _embed(net, "relu_complementarity", sense)
    if box is not None:
        lo, hi = box
        m.subject_to(nn.inputs >= lo, name="box_lo")
        m.subject_to(nn.inputs <= hi, name="box_hi")
    if extra is not None:
        extra(m, nn)
    res = solve_mpec(m, nn.relations, method="native")
    return m, nn, res


def test_native_lowering_is_exact_bilinear_rows():
    m, nn = _embed(_relu_net(hidden=(3,)), "relu_complementarity")
    from discopt.mpec import RELAXING_METHODS, native_pair_columns, reformulate_native

    reformulate_native(m, nn.relations)
    assert "native" not in RELAXING_METHODS
    assert all(r.lowering_in(m) == "native" for r in nn.relations)
    assert sum((c.name or "").endswith("_native") for c in m._constraints) == len(nn.relations)
    cols = native_pair_columns(m)
    assert len(cols) == len(nn.relations)
    # the columns are the y / s elements, in the model's flat layout
    off, names = 0, {}
    for v in m._variables:
        for k in range(v.size):
            names[off + k] = (v.name, k)
        off += v.size
    for (fc, gc), rel in zip(cols, nn.relations):
        assert names[fc] == ("nn_z_0", rel.index[0])
        assert names[gc][0] == rel.g.name


def test_native_lowering_refuses_non_variable_or_negative_operands():
    from discopt.mpec import complementarity, reformulate_native

    m = Model("bad_native")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=-1, ub=1)
    with pytest.raises(ValueError, match="single variable element"):
        reformulate_native(m, [complementarity(x + 0.0, x)])
    with pytest.raises(ValueError, match="lb -1"):
        reformulate_native(m, [complementarity(x, y)])
    assert not m._constraints, "a refused lowering must emit no row"


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("kw,sense", _NETS)
def test_native_matches_relu_bigm(kw, sense, flag, monkeypatch):
    net = _relu_net(**kw)
    mb, _ = _embed(net, "relu_bigm", sense)
    rb = mb.solve()
    assert rb.gap_certified
    _, nn, rc = _solve_native(net, sense, flag, monkeypatch)
    assert rc.status == "optimal" and rc.gap_certified
    assert rc.objective == pytest.approx(rb.objective, abs=1e-6, rel=1e-6)
    # certificate invariant against the oracle
    if sense == "min":
        assert rc.bound <= rb.objective + 1e-6
    else:
        assert rc.bound >= rb.objective - 1e-6
    x_in = np.asarray(rc.x["nn_input"], dtype=np.float64)
    assert float(net.forward(x_in)[0]) == pytest.approx(rc.objective, abs=1e-5)
    stats = rc.solver_stats or {}
    if flag == "1":
        assert "tree/compl_branches" in stats, "the branch was never armed"
    else:
        assert "tree/compl_branches" not in stats


def test_compl_branch_fires_on_a_net_that_needs_branching(monkeypatch):
    """Probe that the dichotomy actually ran (CLAUDE.md §6). One net whose root
    leaves a violated pair, under a deterministic node budget: firing is decided
    at the first branching, so the budget does not weaken the probe."""
    monkeypatch.setenv("DISCOPT_COMPL_BRANCH", "1")
    m, nn = _embed(_relu_net(seed=11, n_in=2, hidden=(5, 4)), "relu_complementarity", "max")
    rc = solve_mpec(m, nn.relations, method="native", max_nodes=8, deterministic=True)
    assert (rc.solver_stats or {}).get("tree/compl_branches", 0) > 0


def test_native_lowering_cuts_no_feasible_point():
    """Feasible-point sampling: every forward-pass point satisfies every row and
    bound of the natively lowered model."""
    from discopt.mpec import reformulate_native
    from discopt.validation.feasibility import verify_point
    from discopt.warm_start import validate_initial_solution

    net = _relu_net(seed=11, n_in=2, hidden=(5, 4))
    m, nn = _embed(net, "relu_complementarity")
    starts = nn.local_starts(200, seed=3)
    reformulate_native(m, nn.relations)
    n_ok = 0
    for st in starts:
        x = validate_initial_solution(m, st)
        assert verify_point(m, x).ok
        n_ok += 1
    assert n_ok == 200


_BOXES = [
    (np.array([-1.0, -1.0]), np.array([0.0, 0.0])),
    (np.array([0.0, -1.0]), np.array([1.0, 0.0])),
    (np.array([-0.5, 0.2]), np.array([0.5, 1.0])),
]


@pytest.mark.parametrize("box", _BOXES, ids=["q3", "q4", "strip"])
def test_compl_branch_bound_on_fixed_boxes(box, monkeypatch):
    """Differential bound test: on a fixed input sub-box the flag-ON certified
    bound never exceeds the box's true optimum (relu_bigm on the same box), and
    the flag-ON objective is the box optimum. (Flag OFF on the same models is
    covered by ``test_native_matches_relu_bigm``.)"""
    net = _relu_net(seed=11, n_in=2, hidden=(5, 4))

    def oracle():
        mb, nnb = _embed(net, "relu_bigm", "min")
        mb.subject_to(nnb.inputs >= box[0], name="box_lo")
        mb.subject_to(nnb.inputs <= box[1], name="box_hi")
        r = mb.solve()
        assert r.gap_certified
        return r.objective

    opt = oracle()
    _, _, on = _solve_native(net, "min", "1", monkeypatch, box=box)
    assert on.gap_certified
    assert on.bound <= opt + 1e-6
    assert on.objective == pytest.approx(opt, abs=1e-6, rel=1e-6)


def _input_rows(m, x):
    """Bound the inputs by model ROWS, not by the network's input box."""
    m.subject_to(x >= -1, name="in_lo")
    m.subject_to(x <= 1, name="in_hi")
    m.subject_to(x[0] + x[1] + x[2] <= 1, name="budget")


def test_compl_branch_certifies_without_input_bounds(monkeypatch):
    """The network declares no input box (relu_bigm refuses it); the inputs are
    bounded only by the surrounding model's rows. The native lowering with the
    branch certifies, and matches relu_bigm given the box explicitly."""
    net_free = _relu_net(seed=7, n_in=3, hidden=(3,), with_bounds=False)
    with pytest.raises(ValueError, match="input_bounds"):
        _embed(net_free, "relu_bigm")
    _, _, rc = _solve_native(
        net_free, "min", "1", monkeypatch, extra=lambda m, nn: _input_rows(m, nn.inputs)
    )
    assert rc.status == "optimal" and rc.gap_certified
    assert (rc.solver_stats or {}).get("tree/compl_branches", 0) > 0

    boxed = NetworkDefinition(net_free.layers, input_bounds=(-np.ones(3), np.ones(3)))
    mb, nnb = _embed(boxed, "relu_bigm", "min")
    _input_rows(mb, nnb.inputs)
    rb = mb.solve()
    assert rb.gap_certified
    assert rc.objective == pytest.approx(rb.objective, abs=1e-6, rel=1e-6)
    assert rc.bound <= rb.objective + 1e-6


def test_compl_branch_sound_on_an_unbounded_relaxation(monkeypatch):
    """Free inputs with a quadratic penalty: the LP relaxation cannot bound
    ``0.1*||x||^2`` over a free ``x``, so this does not certify within a node
    budget (measured on this 3-3-1 net: 10 nodes -> feasible -1.958945, no dual
    bound; on a 3-4-3-1 net 163 nodes / 60 s still gave none). It checks the
    soundness of whatever comes back, which a larger budget does not strengthen.
    Whatever it returns must stay sound: no certificate on a wrong value, no
    bound above the true optimum (relu_bigm on a box wide enough to hold it)."""
    net_free = _relu_net(seed=7, n_in=3, hidden=(3,), with_bounds=False)

    def penalty(m, nn):
        m.minimize(dm.sum(nn.outputs) + 0.1 * dm.sum(nn.inputs * nn.inputs))

    monkeypatch.setenv("DISCOPT_COMPL_BRANCH", "1")
    m, nn = _embed(net_free, "relu_complementarity")
    penalty(m, nn)
    rc = solve_mpec(m, nn.relations, method="native", max_nodes=10, deterministic=True)

    B = 30.0
    wide = NetworkDefinition(net_free.layers, input_bounds=(-B * np.ones(3), B * np.ones(3)))
    mb, nnb = _embed(wide, "relu_bigm", "min")
    mb.minimize(dm.sum(nnb.outputs) + 0.1 * dm.sum(nnb.inputs * nnb.inputs))
    rb = mb.solve()
    assert rb.gap_certified
    assert np.max(np.abs(np.asarray(rb.x["nn_input"]))) < 0.9 * B, "wide box must be inactive"
    # The oracle is certified to an interval [rb.bound, rb.objective]: a primal
    # point may not lie below its bound, a dual bound may not lie above its
    # incumbent. (Measured on a 3-4-3-1 net: the native incumbent -11.5585181
    # sat below the oracle's -11.5585034, inside the oracle's certified gap 1.3e-6.)
    tol = 1e-7 * max(1.0, abs(rb.objective))
    checks = 0
    if rc.objective is not None:
        assert rc.objective >= rb.bound - tol
        x_in = np.asarray(rc.x["nn_input"], dtype=np.float64)
        true_obj = float(net_free.forward(x_in)[0]) + 0.1 * float(x_in @ x_in)
        assert true_obj == pytest.approx(rc.objective, abs=1e-6)
        checks += 1
    if rc.bound is not None:
        assert rc.bound <= rb.objective + tol
        checks += 1
    if rc.gap_certified:
        assert rc.bound <= rb.objective + tol and rc.objective >= rb.bound - tol
        checks += 1
    assert checks > 0, "the solve returned nothing to check"


def test_multistart_time_limit_caps_inflight_start():
    """The budget bounds the run, not only the start of each start: every
    stage and polish NLP is capped by the remaining time (#2167 H1 measured a
    367 s start against a 120 s budget before the cap)."""
    import time

    # He-initialised 4-32-32-1 net (the H1 panel's generator, inlined so the test
    # does not import the benchmark package): ~2.5 s per start, ~50 s uncapped.
    rng = np.random.default_rng(0)
    layers, w = [], 4
    for _ in range(2):
        layers.append(
            DenseLayer(
                rng.standard_normal((w, 32)) * np.sqrt(2.0 / w),
                0.1 * rng.standard_normal(32),
                Activation.RELU,
            )
        )
        w = 32
    layers.append(
        DenseLayer(rng.standard_normal((w, 1)) / np.sqrt(w), np.zeros(1), Activation.LINEAR)
    )
    net = NetworkDefinition(layers, input_bounds=(-np.ones(4), np.ones(4)))
    m, nn = _embed(net, "relu_complementarity")
    t0 = time.perf_counter()
    res = nn.solve_local(n_starts=20, seed=0, time_limit=3.0)
    wall = time.perf_counter() - t0
    assert wall < 3.0 + 8.0, f"budget 3 s, ran {wall:.1f} s"
    assert 1 <= len(res.multistart) < 20
    assert not res.gap_certified
