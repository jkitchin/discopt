"""Pin the issue #721 direction-#1 finding (post-#707): on ex1252's canonical
loosest node, the dual bound is inert to every available relaxation lever, and the
wall is the objective coupling, not the cubic cost rows direction #1 targets.

Entry experiment + root cause: ``docs/dev/performance-plan.md`` §6 (2026-07-18);
reproduced by ``discopt_benchmarks/scripts/ex1252_piecewise_lever_probe.py``.

With #707's integer-multilinear reform applied, the reformed-ex1252 loosest-node
bound equals the objective's constant term (``6329.03·x0·x3·x18 = 12658.06``) and
does not move under RLT / level-1 RLT / PSD / superposition — because the reformed
``x15·(x0·x3·x18)`` objective aux relaxes to its lower bound (the relaxed ``x15`` is
nonzero, yet its ``1800·x15`` cost contributes 0). Tightening the cubic rows that
*define* ``x15`` (direction #1) cannot lift the bound while that coupling is loose.
These tests lock the result so the falsified-here direction is not silently
re-attempted, and double as a soundness pin (no lever ever reports a bound *above*
the true in-node optimum).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("DISCOPT_INCREMENTAL_MC", "0")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.integer_product_reform import reformulate_integer_multilinear
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

pytestmark = pytest.mark.relaxation

CONST_BOUND = 6329.03 * 2.0  # 12658.06 — the objective constant term at the node
TOL = 1e-2
LINE1 = {18: 1, 36: 1, 21: 1, 19: 0, 20: 0, 37: 0, 38: 0, 22: 0, 23: 0}


def _nl_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        for sub in ("python/tests/data/minlplib", "tests/data/minlplib"):
            cand = base / sub / "ex1252.nl"
            if cand.exists():
                return cand
    raise FileNotFoundError("ex1252.nl not found")


@pytest.fixture(scope="module")
def loosest_node():
    r = reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))
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
    return r, lb, ub


def _bound(r, lb, ub, **kw):
    res = MccormickLPRelaxer(r, **kw).solve_at_node(lb.copy(), ub.copy())
    assert res.status == "optimal", f"unexpected status {res.status}"
    return float(res.lower_bound)


@pytest.mark.parametrize(
    "opts",
    [
        {},
        {"rlt_cuts": True},
        {"rlt_level1": True},
        {"psd_cuts": True},
        {"superposition": True},
    ],
    ids=["baseline", "rlt_cuts", "rlt_level1", "psd_cuts", "superposition"],
)
def test_every_lever_inert_at_loosest_node(loosest_node, opts):
    """No available strengthener lifts the bound off the objective-constant floor.

    Also a soundness pin: the bound never exceeds the true in-node optimum
    (128893.74) — an over-tightening would be unsound.
    """
    r, lb, ub = loosest_node
    b = _bound(r, lb, ub, **opts)
    assert b == pytest.approx(CONST_BOUND, abs=TOL), (
        f"a lever moved the bound ({opts}): {b} vs floor {CONST_BOUND}. "
        "If this changed, a coupling-relaxation lever landed — re-run the entry "
        "experiment and update performance-plan.md §6."
    )
    assert b <= 128893.74 + 1e-3, "bound above the true optimum is unsound"


def test_root_cause_x15_coupling_relaxes_to_lower_bound(loosest_node):
    """The wall is the objective coupling: the bound is the constant term, yet the
    relaxed x15 is nonzero — the 1800*x15 cost contributes 0 because the reformed
    x15*(x0*x3*x18) aux relaxes to its lower bound. Tightening the cubic rows that
    define x15 (direction #1) cannot help while this coupling is loose.
    """
    r, lb, ub = loosest_node
    res = MccormickLPRelaxer(r).solve_at_node(lb.copy(), ub.copy())
    assert res.status == "optimal"
    assert float(res.lower_bound) == pytest.approx(CONST_BOUND, abs=TOL)
    # x15 is genuinely nonzero in the relaxed point, so the zero cost contribution
    # is the coupling aux relaxing down — not x15 itself being driven to 0.
    assert float(res.x[15]) > 1.0, "expected a nonzero relaxed x15 (coupling, not x15, is loose)"
