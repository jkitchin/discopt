"""Issue #721 — the objective-coupling RLT (default-OFF
``DISCOPT_MULTILINEAR_COUPLING_RLT``, on top of #707's integer-multilinear reform).

Entry experiment + rationale: ``docs/dev/performance-plan.md`` §6 (2026-07-18).

The #707 reform lifts ex1252's dual off the structural floor but the objective's
``x15·(x0·x3·x18)`` coupling still relaxes to 0 at a line-selected node (the cubic
rows force ``x15 ≥ 12.44``, yet the big-M product envelope lets the ``1800·x15`` cost
contribute nothing — pinning the loosest-node bound at the objective constant
12658.06). The coupling RLT multiplies each integer factor's bit-linking equality
and each AND hull by the (non-negative) continuous factor, tying the disaggregated
products back to ``x15``. This lifts the loosest-node bound to ~57435 — a large,
**sound** tightening (the RLT rows are products of valid identities/inequalities, so
they never cut a feasible point).

These tests lock: (1) the OFF path is byte-identical to #707 (same bound, same var
count); (2) the ON path lifts the node bound to ~57435 and is sound (≤ the true
optimum); (3) no feasible point is cut. The flag stays default-OFF — in a full
time-limited solve the extra per-node bilinears currently cost enough throughput to
be net-negative on the *global* dual (see §6), so graduation waits on deep-node
gating; here we pin soundness and the node-level gain.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

pytestmark = [pytest.mark.relaxation, pytest.mark.correctness]

OFF_BOUND = 6329.03 * 2.0  # 12658.06 — objective constant; #707 baseline at the node
ON_BOUND = 57434.96  # coupling-RLT lifted bound (12658.06 + 3600·min_x15, min_x15≈12.44)
EX1252_OPT = 128893.74
LINE1 = {18: 1, 36: 1, 21: 1, 19: 0, 20: 0, 37: 0, 38: 0, 22: 0, 23: 0}


def _nl_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        for sub in ("python/tests/data/minlplib", "tests/data/minlplib"):
            cand = base / sub / "ex1252.nl"
            if cand.exists():
                return cand
    raise FileNotFoundError("ex1252.nl not found")


def _loosest_node_bound(monkeypatch, coupling_rlt: bool):
    # monkeypatch reverts the env change automatically (conftest guards against
    # leaked DISCOPT_* mutations under xdist).
    monkeypatch.setenv("DISCOPT_MULTILINEAR_COUPLING_RLT", "1" if coupling_rlt else "0")
    r = reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))
    n_vars = sum(v.size for v in r._variables)
    lb, ub = flat_variable_bounds(r)
    lb = np.asarray(lb, float).copy()
    ub = np.asarray(ub, float).copy()
    for i, v in LINE1.items():
        lb[i] = ub[i] = float(v)
    nc = obbt_tighten_root(r, lb.copy(), ub.copy(), rounds=5, time_limit_per_lp=0.3)
    lb, ub = nc.lb.copy(), nc.ub.copy()
    lb[0] = ub[0] = 2.0
    lb[24] = ub[24] = 0.0
    lb[25] = ub[25] = 1.0  # x0 = 2
    lb[3] = ub[3] = 1.0
    lb[30] = ub[30] = 1.0
    lb[31] = ub[31] = 0.0  # x3 = 1
    res = MccormickLPRelaxer(r).solve_at_node(lb.copy(), ub.copy())
    assert res.status == "optimal", f"status {res.status}"
    return float(res.lower_bound), n_vars


def test_off_path_is_707_baseline(monkeypatch):
    """Flag OFF: byte-identical to #707 — bound at the objective-constant floor, no
    extra columns (the coupling RLT adds nothing when disabled)."""
    bound, n_vars = _loosest_node_bound(monkeypatch, coupling_rlt=False)
    assert bound == pytest.approx(OFF_BOUND, abs=1e-2)
    assert n_vars == 90, f"OFF path must not add columns; got {n_vars}"


def test_on_path_lifts_bound_soundly(monkeypatch):
    """Flag ON: the coupling RLT lifts the loosest-node bound ~4.5x, and it stays
    sound — a valid dual bound never exceeds the true optimum."""
    bound, n_vars = _loosest_node_bound(monkeypatch, coupling_rlt=True)
    assert bound == pytest.approx(ON_BOUND, rel=1e-3), f"expected ~{ON_BOUND}, got {bound}"
    assert bound > OFF_BOUND + 1000.0, "coupling RLT must materially lift the bound"
    assert bound <= EX1252_OPT + 1e-2, f"UNSOUND: bound {bound} > optimum {EX1252_OPT}"
    assert n_vars > 90, "the RLT products add columns when enabled"


def test_coupling_rlt_does_not_cut_the_optimum(monkeypatch):
    """Soundness: the true optimal point (bound ≤ opt) is never cut — the ON bound is
    a valid lower bound, strictly below the instance optimum (no over-tightening)."""
    on_bound, _ = _loosest_node_bound(monkeypatch, coupling_rlt=True)
    assert on_bound <= EX1252_OPT + 1e-2
    off_bound, _ = _loosest_node_bound(monkeypatch, coupling_rlt=False)
    # Monotone: adding valid RLT rows can only tighten (raise) the bound.
    assert on_bound >= off_bound - 1e-6, "valid RLT rows must never loosen the bound"
