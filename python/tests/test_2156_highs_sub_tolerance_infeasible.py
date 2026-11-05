"""#2156: the HiGHS MILP route must not certify ``infeasible`` on a feasible model.

On a ReLU network with tiny weights, the unmerged ``relu_extended`` encoding has
continuous columns with ``0 < xu - xl < 1e-6``. HiGHS's MIP labels the model
``kInfeasible``, and every route cross-check agrees, because all of them run the
same MIP code. Measured on 30 such 2-32-32-1 networks: 8 came back
``infeasible`` with ``gap_certified=True``.

Checked here:

* End to end: that model, with the #2156 partition merge switched off, is not
  certified infeasible. A point built from ``net.forward`` passes
  ``verify_point``. That point is an independent oracle that the model is
  feasible.
* The guard's scope: it withdraws a HiGHS-only label, leaves a Farkas proof and a
  model without sub-tolerance columns alone, and never touches ``optimal``.
* Canary: on the shipped 13-row sub-model, raw HiGHS still answers
  ``kInfeasible``. The shipped point is feasible on it. When this canary fails,
  HiGHS no longer mislabels this model: re-measure the #2156 panel, then drop
  :func:`~discopt.solvers.lp_milp_highs._sub_tolerance_infeasible_guard` or keep it
  as a documented guard.

Each test counts its comparisons and fails if the count is zero (CLAUDE.md §6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.ml import DenseLayer, NetworkDefinition, NNFormulation
from discopt.ml.formulations import relu_partition as rp
from discopt.modeling import Model
from discopt.solvers import lp_milp_highs as R
from discopt.validation.feasibility import verify_point

DATA = Path(__file__).parent / "data" / "highs_2156_false_infeasible.npz"


def _tiny_weight_net(seed: int, hidden: int) -> NetworkDefinition:
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


def _exact_point(m: Model, net: NetworkDefinition, x_in: np.ndarray) -> np.ndarray:
    """The extended formulation's variables at input ``x_in``, from ``net.forward``."""
    vals = {"nn_input": x_in}
    a = x_in
    n_layers = len(net.layers)
    for k, layer in enumerate(net.layers):
        W, b = np.asarray(layer.weights), np.asarray(layer.biases)
        pre = a @ W + b
        vals[f"nn_zhat_{k}"] = pre
        if k == n_layers - 1:
            vals["nn_output"] = pre
            break
        vals[f"nn_z_{k}"] = np.maximum(pre, 0.0)
        for j in range(pre.size):
            on = float(pre[j] > 0)
            vals[f"nn_q_{k}_{j}"] = np.array(on)
            parts = rp.partition_indices(W[:, j], None)
            vals[f"nn_zb_{k}_{j}"] = np.array([float(W[p, j] @ a[p]) * on for p in parts])
        a = np.maximum(pre, 0.0)
    return np.concatenate([np.atleast_1d(vals[v.name]).ravel() for v in m._variables])


#: Big-M's certified optimum on ``_tiny_weight_net(0, 32)`` (measured, highspy 1.15.1).
SEED0_OPTIMUM = 0.04328811180858727


def test_unmerged_extended_model_is_not_certified_infeasible(monkeypatch):
    """Before #2156 this returned ``infeasible`` with ``gap_certified=True``.

    Feasibility is shown without a second solve: the forward pass at any input is an
    exact point of the model (two inputs are checked). The big-M reference solve was
    dropped to fit the fast-lane time budget; its optimum is :data:`SEED0_OPTIMUM`.
    """
    monkeypatch.setattr(rp, "NEGLIGIBLE_PARTITION_RTOL", 0.0)
    monkeypatch.setattr(rp, "NEGLIGIBLE_PARTITION_ATOL", 0.0)
    net = _tiny_weight_net(0, 32)
    n = 0
    m = Model("ext")
    nn = NNFormulation(m, net, strategy="relu_extended")
    nn.formulate()
    m.minimize(nn.outputs[0])
    for x_in in (np.zeros(2), np.array([1.0, -2.0])):
        vr = verify_point(m, _exact_point(m, net, x_in), with_objective=True)
        assert vr.ok, vr.reason
        n += 1
        assert vr.objective == pytest.approx(float(net.forward(x_in[None, :]).ravel()[0]))
        n += 1

    r = m.solve(time_limit=120, threads=1, deterministic=True)
    assert r.status != "infeasible", (r.status, r.gap_certified)
    n += 1
    assert not (r.gap_certified and r.status != "optimal")
    n += 1
    if r.status == "optimal":
        assert r.objective == pytest.approx(SEED0_OPTIMUM, rel=1e-4, abs=1e-6)
        n += 1
    assert n >= 6


def _sf_with_width(width: float, integer_col: bool = False) -> R.StdForm:
    A = sp.csc_matrix(np.array([[1.0, 1.0]]))
    xl = np.array([0.0, 0.0])
    xu = np.array([width, 1.0])
    return R.StdForm.from_arrays(np.zeros(2), A, [0.5], xl, xu, 0.0, [0] if integer_col else [1])


def test_sub_tolerance_columns_counts_only_open_narrow_continuous():
    n = 0
    for width, integer_col, expect in [
        (5e-7, False, 1),  # narrow continuous: counted
        (0.0, False, 0),  # exactly fixed: not counted
        (2e-6, False, 0),  # wide enough
        (5e-7, True, 0),  # narrow integer column: not counted
    ]:
        assert R.sub_tolerance_columns(_sf_with_width(width, integer_col)) == expect
        n += 1
    sf = _sf_with_width(1.0)
    sf.xu[0] = R.INF
    assert R.sub_tolerance_columns(sf) == 0  # open side
    n += 1
    assert n == 5


@pytest.mark.parametrize(
    "status,provenance,width,withdrawn",
    [
        ("infeasible", "highs", 5e-7, True),
        ("infeasible", "highs-root-feasible", 5e-7, True),
        ("infeasible", "farkas-root-lp", 5e-7, False),  # the route's own proof stands
        ("infeasible", "highs", 1.0, False),  # no sub-tolerance column
        ("optimal", None, 5e-7, False),  # the guard is about infeasible labels only
    ],
)
def test_guard_scope(status, provenance, width, withdrawn):
    labels = {} if provenance is None else {"milp/infeasible_provenance": provenance}
    out = R.HighsOutcome(status, gap_certified=True, labels=dict(labels))
    got = R._sub_tolerance_infeasible_guard(_sf_with_width(width), out)
    n = 0
    if withdrawn:
        assert (got.status, got.gap_certified, got.bound) == ("error", False, None)
        n += 1
        assert "milp/infeasible_provenance" not in got.labels
        assert got.labels.get("milp/certificate") == "declined"
        assert "#2156" in got.message
        n += 1
    else:
        assert (got.status, got.gap_certified) == (status, True)
        n += 1
        assert got.labels.get("milp/infeasible_provenance") == provenance
        n += 1
    assert n == 2


def _canary_model():
    d = np.load(DATA)
    A = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    sf = R.StdForm.from_arrays(np.zeros(A.shape[1]), A, d["b"], d["xl"], d["xu"], 0.0, d["int_idx"])
    return sf, d["x_feasible"]


def test_canary_highs_still_mislabels_the_13_row_submodel():
    """Fails when the installed HiGHS stops answering ``kInfeasible`` here (see module)."""
    highspy = pytest.importorskip("highspy")
    sf, x = _canary_model()
    n = 0
    assert R.feasibility_problem(x, sf, check_integrality=True) is None
    n += 1
    assert R.sub_tolerance_columns(sf) > 0
    n += 1
    highspy.Highs.resetGlobalScheduler(True)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "off")
    st, why = R._pass_model(h, highspy, sf, True, 0.0)
    assert st == highspy.HighsStatus.kOk, why
    h.run()
    assert R._status_name(h) == "kInfeasible", (
        "HiGHS no longer labels the #2156 sub-model infeasible: re-measure the #2156 "
        "panel and drop or document _sub_tolerance_infeasible_guard"
    )
    n += 1
    assert n == 3
