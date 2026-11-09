"""Negligible-range partitions in the ReLU partition / extended MILPs (#2156).

Trained networks carry many weights of 1e-6 to 1e-8. With one partition per input
(``relu_extended``), each such input got an auxiliary ``zb`` whose whole range
``[LB_p, UB_p]`` was below HiGHS's feasibility tolerance. HiGHS then answered
``kInfeasible`` (presolve on and off) on feasible models, or a wrong optimum. On a
synthetic 30-network panel (2-32-32-1, 70 % of hidden weights scaled by
10**U(-8, -4)) the old encoding gave 0/30 certified optima for ``relu_extended``,
with 8 certified-``infeasible`` answers, and 0/30 for ``relu_partition`` with
P = 4. With the merge both are 30/30 certified and equal to big-M.

Checked here:

* :func:`merge_negligible_partitions` keeps a cover of the inputs, folds into the
  widest partition, and is a no-op at ``rtol=0``.
* Structure: no auxiliary of the emitted model has a partition range at or below
  ``NEGLIGIBLE_PARTITION_RTOL`` times its neuron's range or below
  ``NEGLIGIBLE_PARTITION_ATOL``.
* The issue's own trained network (``data/relu_2156_issue_net.npz``): all three
  ReLU MILPs certify the same optimum, about -6.3749.
* End to end: ``relu_extended`` and ``relu_partition`` (P = 4) certify the big-M
  optimum on tiny-weight networks, and the optimal input reproduces the objective
  through ``net.forward`` (an independent oracle).

Each test counts its comparisons and fails if the count is zero (CLAUDE.md §6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from discopt.ml import DenseLayer, NetworkDefinition, NNFormulation
from discopt.ml.formulations import relu_partition as rp
from discopt.ml.formulations.relu_partition import (
    NEGLIGIBLE_PARTITION_ATOL,
    NEGLIGIBLE_PARTITION_RTOL,
    merge_negligible_partitions,
    partition_indices,
)
from discopt.modeling import Model


def _tiny_weight_net(seed: int, hidden: int) -> NetworkDefinition:
    """2-h-h-1 ReLU net, 70 % of hidden weights shrunk to 1e-8..1e-4 (trained-like)."""
    g = np.random.default_rng(seed)
    sizes = [2, hidden, hidden, 1]
    layers = []
    for i in range(3):
        W = g.normal(0, 1 / np.sqrt(sizes[i]), (sizes[i], sizes[i + 1]))
        b = g.normal(0, 0.1, sizes[i + 1])
        if i < 2:
            mask = g.random(W.shape) < 0.7
            W[mask] *= 10 ** g.uniform(-8, -4, mask.sum())
        layers.append(DenseLayer(W, b, "relu" if i < 2 else "linear"))
    return NetworkDefinition(layers, input_bounds=(np.full(2, -3.0), np.full(2, 3.0)))


def _solve(net, strategy, **kw):
    m = Model("nn")
    nn = NNFormulation(m, net, strategy=strategy, **kw)
    nn.formulate()
    m.minimize(nn.outputs[0])
    return m, m.solve(time_limit=120, threads=1, deterministic=True)


def test_merge_keeps_a_cover_and_folds_into_widest():
    n = 0
    tr = np.array([5.0, 1e-9, 3.0, 2e-9, 1e-12, 0.5])
    parts = [np.array([i]) for i in range(tr.size)]
    out = merge_negligible_partitions(parts, tr, rtol=1e-6)
    flat = np.sort(np.concatenate(out))
    assert flat.tolist() == list(range(tr.size))
    n += 1
    assert len(out) == 3
    n += 1
    # The three negligible inputs land with input 0, the widest.
    widest = next(p for p in out if 0 in p.tolist())
    assert sorted(widest.tolist()) == [0, 1, 3, 4]
    n += 1
    # Every remaining partition is wider than the threshold.
    total = tr.sum()
    for p in out:
        assert tr[p].sum() > 1e-6 * total
        n += 1
    assert n >= 6


def test_merge_noop_cases_and_validation():
    n = 0
    tr = np.array([1.0, 1e-12, 2.0])
    parts = [np.array([i]) for i in range(3)]
    for got in (
        merge_negligible_partitions(parts, tr, rtol=0.0, atol=0.0),
        merge_negligible_partitions([np.array([0, 1, 2])], tr),
        merge_negligible_partitions(parts, np.array([1.0, 1.0, 1.0])),
    ):
        assert sum(p.size for p in got) == 3
        n += 1
    assert len(merge_negligible_partitions(parts, tr, rtol=0.0, atol=0.0)) == 3
    n += 1
    # All-zero range (fixed inputs): one partition, i.e. big-M.
    assert len(merge_negligible_partitions(parts, np.zeros(3))) == 1
    n += 1
    with pytest.raises(ValueError):
        merge_negligible_partitions(parts, tr, rtol=-1.0)
    n += 1
    assert n == 6


def test_absolute_floor_merges_narrow_partitions_of_a_narrow_neuron():
    """Review of #2158: a neuron of total range 2.5e-6 in pieces of 9e-7, 9e-7 and
    7e-7 passes the relative test (each > 1e-4 * 2.5e-6), but every piece is below
    the 1e-6 solver tolerance."""
    n = 0
    tr = np.array([9e-7, 9e-7, 7e-7])
    parts = [np.array([i]) for i in range(3)]
    assert len(merge_negligible_partitions(parts, tr, atol=0.0)) == 3  # relative test only
    n += 1
    assert len(merge_negligible_partitions(parts, tr)) == 1  # all below the floor: big-M
    n += 1
    # A wide partition absorbs the narrow ones and is kept.
    tr2 = np.array([1.0, 5e-7, 2e-6])
    out = merge_negligible_partitions(parts, tr2, rtol=1e-9)
    assert [sorted(p.tolist()) for p in out] == [[0, 1], [2]]
    n += 1
    assert n == 3


def _load_issue_net() -> NetworkDefinition:
    d = np.load(Path(__file__).parent / "data" / "relu_2156_issue_net.npz")
    k = int(d["n_layers"])
    layers = [
        DenseLayer(d[f"W{i}"], d[f"b{i}"], "relu" if i < k - 1 else "linear") for i in range(k)
    ]
    return NetworkDefinition(layers, input_bounds=(d["lo"], d["hi"]))


@pytest.mark.parametrize(
    "strategy,kw",
    [
        ("relu_bigm", {}),
        ("relu_partition", {}),
        ("relu_partition", {"n_partitions": 4}),
        ("relu_extended", {}),
    ],
)
def test_issue_network_certifies_the_bigm_optimum(strategy, kw, monkeypatch):
    """The #2156 reproducer: ``relu_extended`` was ``feasible`` with gap 0.69 and
    relu_partition P = 4 drifted; all now certify the optimum (-6.3749 to 1e-4).

    The subject is the formulation, so the MILP is solved where it was written to be,
    the HiGHS route (the optional ``discopt[highs]`` extra). On the Rust default
    (PR #2196) the same models certify too, but slowly: ``relu_extended`` 62 s and
    ``relu_partition`` P = 4 13 s on a dev machine (91 s / 37 s in CI) against 4 s and
    3 s on HiGHS; recorded in ``docs/dev/highs-parity-d2-gap-plan.md`` §8.
    """
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    net = _load_issue_net()
    _, r = _solve(net, strategy, **kw)
    n = 0
    assert r.status == "optimal", (r.status, r.objective, r.gap)
    n += 1
    assert r.objective == pytest.approx(-6.3749, rel=1e-4)
    n += 1
    x_in = np.asarray(r.x["nn_input"], dtype=float)
    assert float(net.forward(x_in[None, :]).ravel()[0]) == pytest.approx(
        r.objective, rel=1e-4, abs=1e-6
    )
    n += 1
    assert n == 3


def _build(net, cls, **kw):
    m = Model("nn")
    f = cls(m, net, "nn", None, **kw)
    f.build()
    zb = {v.name: v for v in m._variables if "_zb_" in v.name}
    return f, zb


@pytest.mark.parametrize(
    "cls,kw",
    [(rp.ReluExtendedFormulation, {}), (rp.ReluPartitionFormulation, {"n_partitions": 4})],
)
def test_no_auxiliary_of_negligible_range(cls, kw):
    """Every emitted partition is wider than ``rtol`` times its neuron's range, and
    the merge fired on this class (else the end-to-end test proves nothing)."""
    net = _tiny_weight_net(0, 16)
    f, zb = _build(net, cls, **kw)
    n = n_merged = 0
    for rec in f.relu_layers:
        width = np.abs(rec.weights) * (rec.in_ub - rec.in_lb)[:, None]
        for j, q in enumerate(rec.binaries):
            if q is None:
                continue
            w = rec.weights[:, j]
            raw = partition_indices(w, kw.get("n_partitions"))
            got = merge_negligible_partitions(raw, width[:, j])
            n_merged += len(raw) - len(got)
            name = f"nn_zb_{rec.layer_index}_{j}"
            if len(got) <= 1:
                assert name not in zb  # emitted as big-M
                n += 1
                continue
            assert zb[name].size == len(got)
            n += 1
            total = float(width[:, j].sum())
            for p in got:
                r_p = float(width[p, j].sum())
                assert r_p > NEGLIGIBLE_PARTITION_RTOL * total
                assert r_p >= NEGLIGIBLE_PARTITION_ATOL
                n += 1
    assert n > 0
    assert n_merged > 0


def test_rtol_zero_restores_one_partition_per_input(monkeypatch):
    """``NEGLIGIBLE_PARTITION_RTOL = ATOL = 0`` emits the unmerged P = n_in hull."""
    monkeypatch.setattr(rp, "NEGLIGIBLE_PARTITION_RTOL", 0.0)
    monkeypatch.setattr(rp, "NEGLIGIBLE_PARTITION_ATOL", 0.0)
    f, zb = _build(_tiny_weight_net(0, 16), rp.ReluExtendedFormulation)
    n = 0
    for rec in f.relu_layers:
        for j, q in enumerate(rec.binaries):
            if q is None:
                continue
            nnz = int(np.count_nonzero(rec.weights[:, j]))
            if nnz > 1:
                assert zb[f"nn_zb_{rec.layer_index}_{j}"].size == nnz
                n += 1
    assert n > 0


@pytest.mark.parametrize("seed,hidden", [(0, 32), (0, 16)])
@pytest.mark.parametrize(
    "strategy,kw", [("relu_extended", {}), ("relu_partition", {"n_partitions": 4})]
)
def test_tiny_weight_net_certifies_bigm_optimum(seed, hidden, strategy, kw, monkeypatch):
    # HiGHS is the opt-in MILP engine since PR #2196; this test needs a certifying
    # oracle engine. The Rust engine's shortfall here is a recorded MILP-gap target.
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")
    """Before #2156, (0, 32) relu_extended returned a certified ``infeasible`` on a
    feasible model; (0, 16) returned ``feasible`` (uncertified)."""
    net = _tiny_weight_net(seed, hidden)
    _, rb = _solve(net, "relu_bigm")
    m, r = _solve(net, strategy, **kw)
    n = 0
    assert rb.status == "optimal"
    n += 1
    assert r.status == "optimal", (r.status, r.objective, rb.objective)
    n += 1
    assert r.objective == pytest.approx(rb.objective, rel=1e-4, abs=1e-6)
    n += 1
    x_in = np.asarray(r.x["nn_input"], dtype=float)
    assert float(net.forward(x_in[None, :]).ravel()[0]) == pytest.approx(
        r.objective, rel=1e-4, abs=1e-6
    )
    n += 1
    assert n == 4
