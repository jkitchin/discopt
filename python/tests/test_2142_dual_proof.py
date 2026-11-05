"""#2142: learned dual-proof (Farkas) rows in the native spatial kernel.

``DISCOPT_DUAL_PROOF_PROPAGATION`` (``SpatialTreeConfig::dual_proof_propagation``):
on a certified-infeasible node LP the kernel aggregates the verified Farkas ray over
the GLOBAL rows only (or, when that ray does not cut the node, the ray of the
global-rows feasibility LP on the same box), keeps the row if it cuts the node, and
propagates the tree-global pool at every later node. Never an LP row.

Pinned here on GDPlib ``cstr`` big-M's kernel spec (``data/spatial_spec_cstr_bigm_1917.npz``):

* OFF is the tree before the change, bit for bit (no kwarg == ``False``), with every
  counter zero;
* ON learns rows, they tighten boxes, the bound rises, and it stays below the known
  optimum (a learned row that cut a feasible point could push it above).

The Rust side carries the soundness tests (feasible-point sampling on random LPs,
outward rhs rounding): ``bnb::dual_proof::tests``. Measurements:
``docs/dev/performance-plan.md`` section 112.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

R = pytest.importorskip("discopt._rust")

_SPEC = Path(__file__).parent / "data" / "spatial_spec_cstr_bigm_1917.npz"
_CSTR_OPT = 3.0620145717261655
_NODES = 1500


def _spec() -> dict:
    with np.load(_SPEC) as z:
        kw = {k: z[k] for k in z.files if k != "__scalars__"}
        kw.update(json.loads(str(z["__scalars__"])))
    kw.update(max_nodes=_NODES, time_limit_s=1e6, bound_time_extension_s=None)
    return kw


@pytest.fixture(scope="module")
def runs():
    return {
        "default": R.solve_spatial_tree_py(**_spec()),
        "off": R.solve_spatial_tree_py(**_spec(), dual_proof_propagation=False),
        "on": R.solve_spatial_tree_py(**_spec(), dual_proof_propagation=True),
    }


def test_flag_off_is_the_tree_before_it(runs):
    executed = 0
    for k in ("status", "bound", "node_count", "n_lp_solves", "n_undecided"):
        assert runs["default"][k] == runs["off"][k], k
        executed += 1
    assert all(v == 0 for v in runs["off"]["dual_proof"].values()), runs["off"]["dual_proof"]
    executed += 1
    assert executed == 6


def test_cstr_bigm_learns_proofs_that_lift_a_sound_bound(runs):
    on, off = runs["on"], runs["off"]
    dp = on["dual_proof"]
    executed = 0
    assert on["node_count"] == off["node_count"] == _NODES
    executed += 1
    # The mechanism fired (CLAUDE.md §6): rays examined, rows learned, rows used.
    assert dp["rays"] > 0 and dp["learned"] > 0, dp
    assert dp["tightenings"] > 0 and dp["propagations"] > 0, dp
    assert dp["pool_size"] == dp["learned"] - dp["evicted"], dp
    executed += 3
    # Sound: never above the known optimum.
    assert off["bound"] <= _CSTR_OPT + 1e-6
    assert on["bound"] <= _CSTR_OPT + 1e-6
    executed += 2
    # Helpful on the class it was built for (measured at this budget; see section 112).
    assert on["bound"] > off["bound"], (on["bound"], off["bound"])
    executed += 1
    assert executed == 7


def test_python_flag_graduated_default_on_with_opt_out(monkeypatch):
    from discopt.solver import _kernel_dual_proof_enabled

    monkeypatch.delenv("DISCOPT_DUAL_PROOF_PROPAGATION", raising=False)
    assert _kernel_dual_proof_enabled() is True
    monkeypatch.setenv("DISCOPT_DUAL_PROOF_PROPAGATION", "0")
    assert _kernel_dual_proof_enabled() is False
    monkeypatch.setenv("DISCOPT_DUAL_PROOF_PROPAGATION", "1")
    assert _kernel_dual_proof_enabled() is True
