"""The monolithic Rust MILP engine certifies trained ReLU big-M MILPs (#2163).

On trained 2-32-32-1 ReLU networks (hidden weights from ~1e-8 to ~10) the Rust
engine (``DISCOPT_LP_MILP_BACKEND=rust``) handed back hundreds of node LPs
undecided and drained into ``unresolved_floor``: ``feasible`` with a wide gap where
HiGHS certifies the optimum in a second. Two defects, one fixture each:

* ``scaled_numerical`` -- the driver's ``ScaledProblem`` solves ended ``Numerical``
  on the equilibrated matrix (column factors up to 2^16 squeeze the boxes) and,
  unlike every other auto-scaling entry point, had no #649 retry on the original
  matrix, which decides them. Before the fix: ``feasible``, gap 0.13.
* ``snapped_incumbent`` -- a node vertex with a binary at ``1 - 5e-6`` (inside the
  integrality tolerance) verified as computed, but its integral realisation misses
  a big-M row by 3e-6; the driver adopted it, ``verify_point`` refused it, and the
  solve ended ``error``. The driver now verifies the snapped point and, when that
  fails, re-solves the node LP with the integers fixed (a verified incumbent).
* ``cuts_a`` / ``cuts_b`` -- the same networks with native Anderson cuts (#2164),
  which run only on the Rust engine. The cut-augmented node LPs end ``Numerical``
  on the scaled AND the unscaled data (the primal stops at a basis its final audit
  refuses, with phase-1 artificials still basic). ``ScaledProblem`` now also solves
  them with the dual simplex from an all-logical basis, as HiGHS starts. Before:
  ``feasible``, gap 0.39 and 0.087.

Both are 2-12-12-1 sub-networks of network A / B of the issue (the
``net.json`` attachments to #2156 / #2151), stored in
``data/relu_bigm_2163_nets.json``. The oracle is the default HiGHS route on the
same model, whose certificate discopt verifies independently (#1229).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("highspy")

import discopt.modeling as dm  # noqa: E402
from discopt.ml import NNFormulation  # noqa: E402
from discopt.ml.network import Activation, DenseLayer, NetworkDefinition  # noqa: E402

_NETS = json.loads((Path(__file__).parent / "data" / "relu_bigm_2163_nets.json").read_text())


def _solve(name: str, cuts: bool = False):
    d = _NETS[name]
    layers = [(np.array(lay["W"]), np.array(lay["b"])) for lay in d["layers"]]
    net = NetworkDefinition(
        [DenseLayer(W, b, Activation.RELU) for W, b in layers[:-1]]
        + [DenseLayer(*layers[-1], Activation.LINEAR)],
        input_bounds=(np.array(d["lo"]), np.array(d["hi"])),
    )
    m = dm.Model(f"relu_2163_{name}")
    nn = NNFormulation(m, net, strategy="relu_bigm", ideal_cuts=cuts)
    nn.formulate()
    m.minimize(dm.sum(nn.outputs))
    # time_limit is a backstop (each solve measured ~0.3-0.6 s); determinism keeps
    # the node sequence fixed.
    extra = {"cut_callback": nn.cut_callback} if cuts else {}
    return m.solve(time_limit=60, threads=1, deterministic=True, **extra)


@pytest.mark.parametrize(
    ("name", "cuts"),
    [
        ("scaled_numerical", False),
        ("snapped_incumbent", False),
        ("cuts_a", True),
        ("cuts_b", True),
    ],
)
def test_rust_engine_certifies_the_highs_optimum(name, cuts, monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_MILP_BACKEND", raising=False)
    # The oracle is the plain big-M model on HiGHS: cuts never change the optimum.
    ref = _solve(name)
    assert ref.status == "optimal", f"oracle (HiGHS route) did not certify: {ref.status}"
    assert "highs" in str(ref.algorithm_route).lower(), ref.algorithm_route

    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    r = _solve(name, cuts)
    route = str(r.algorithm_route).lower()
    if cuts:
        assert "native-relu-cuts" in route, r.algorithm_route
    else:
        assert "rust milp engine" in route, r.algorithm_route

    checks = 0
    assert r.status == "optimal", (r.status, r.objective, r.bound, r.gap)
    checks += 1
    assert r.gap_certified, (r.gap, r.bound)
    checks += 1
    tol = 1e-4 * (1.0 + abs(ref.objective))
    assert abs(r.objective - ref.objective) <= tol, (r.objective, ref.objective)
    checks += 1
    # Certificate invariant (min sense): the dual bound never crosses the optimum.
    assert r.bound <= ref.objective + tol, (r.bound, ref.objective)
    checks += 1
    assert checks == 4
