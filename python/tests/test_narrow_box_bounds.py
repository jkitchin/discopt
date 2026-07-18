"""#732 Stage 1 — engine hardening on narrow/pinned/crossed node boxes.

Root causes found and fixed (record: ``docs/dev/ex1252-certification-plan.md``
Stage 1; entry experiment: ``discopt_benchmarks/scripts/ex1252_compounding_probe.py``):

1. **Directional-widening bug in the conditioning clamp** (``solve_at_node`` and
   ``sanitize_relaxation_for_conditioning``): a bound crossing the numeric cap was
   mapped to ±inf *by sign*, so a large-**positive lower** bound became ``+inf`` —
   a pinned ``[+inf, +inf)`` box, not a widening. On ex1252 config children the
   ``x6**3`` monomial aux (lb ≥ 1.9e10 ≥ the 1e10 cap on high-speed sub-boxes) hit
   exactly this; the simplex reported the nonsense LP ``unbounded`` and the
   objective-floor fallback collapsed the child bound to 0.0. The fix widens
   directionally (crossing lo → −inf, crossing hi → +inf) — identical on the
   ±sentinel cases the clamp was written for, and a true widening otherwise.

2. **Crossed (empty) node boxes crashed the build** into a diagnostic-free
   ``status="error"``: a sound tightener (OBBT returns ``infeasible=True`` when its
   tightening crosses bounds) hands back a provably empty box; callers that pass it
   to ``solve_at_node`` should get the definitionally correct ``infeasible`` — an
   empty box contains no point. Hair-crossings (float round-off) are instead
   repaired by widening to the enclosing box, which can never produce a false prune.

These tests fail before the fixes and pass after; the ex1252 integration case also
pins soundness (bound ≤ true optimum) and monotone child progress.
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
from discopt._jax.milp_relaxation import (
    MilpRelaxationModel,
    sanitize_relaxation_for_conditioning,
)
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root

pytestmark = [pytest.mark.relaxation, pytest.mark.correctness]

EX1252_OPT = 128893.74
# Feasible line-1-only configuration (indicators + equality-pinned binaries). The
# earlier anchor (LINE1 + x0=2, x3=1) is a config OBBT proves EMPTY at rounds=8;
# the #732 Stage-2 bilinear RLT correctly exposes that at the raw node, so the
# battery anchors on this feasible config instead.
CONFIG_100 = ((18, 36, 1.0), (19, 37, 0.0), (20, 38, 0.0))


def _nl_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        for sub in ("python/tests/data/minlplib", "tests/data/minlplib"):
            cand = base / sub / "ex1252.nl"
            if cand.exists():
                return cand
    raise FileNotFoundError("ex1252.nl not found")


# ---------------------------------------------------------------------------
# 1. Directional widening of the conditioning clamp (unit level)
# ---------------------------------------------------------------------------


def test_conditioning_clamp_widens_directionally():
    """A cap-crossing bound must WIDEN its side (lo→-inf, hi→+inf) — never pin.

    The old sign-based mapping sent a large-positive lower bound to +inf (a
    ``[+inf, +inf)`` box) and a large-negative upper bound to -inf. Both cases
    must now widen; the ±sentinel cases and well-scaled bounds are unchanged.
    """
    model = MilpRelaxationModel(
        c=np.zeros(4),
        A_ub=None,
        b_ub=None,
        bounds=[
            (1.9e10, 2.57e10),  # large-POSITIVE lo (the ex1252 x6^3 aux case)
            (-2.57e10, -1.9e10),  # large-NEGATIVE hi (mirror case)
            (-1e20, 1e20),  # the ±sentinel case the clamp was written for
            (-5.0, 7.0),  # well-scaled: untouched
        ],
    )
    out = sanitize_relaxation_for_conditioning(model)
    (lo0, hi0), (lo1, hi1), (lo2, hi2), (lo3, hi3) = out._bounds
    # Crossing sides widen outward — never a +inf lower / -inf upper bound.
    assert lo0 == -np.inf and hi0 == np.inf
    assert lo1 == -np.inf and hi1 == np.inf
    assert lo2 == -np.inf and hi2 == np.inf
    assert (lo3, hi3) == (-5.0, 7.0)
    for lo, hi in out._bounds:
        assert not (lo == np.inf), "a lower bound of +inf is a pinned, nonsense box"
        assert not (hi == -np.inf), "an upper bound of -inf is a pinned, nonsense box"


# ---------------------------------------------------------------------------
# 2/3. ex1252 config-node integration + empty-box guard
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config_node():
    os.environ["DISCOPT_MULTILINEAR_COUPLING_RLT"] = "1"
    try:
        r = reformulate_integer_multilinear(dm.from_nl(str(_nl_path())))
    finally:
        os.environ.pop("DISCOPT_MULTILINEAR_COUPLING_RLT", None)
    lb, ub = flat_variable_bounds(r)
    lb = np.asarray(lb, float).copy()
    ub = np.asarray(ub, float).copy()
    for ind, sel, v in CONFIG_100:
        lb[ind] = ub[ind] = v
        lb[sel] = ub[sel] = v
    nc = obbt_tighten_root(r, lb.copy(), ub.copy(), rounds=5, time_limit_per_lp=0.3)
    assert not nc.infeasible
    return r, nc.lb.copy(), nc.ub.copy()


def test_high_speed_child_boxes_certify_real_bounds(config_node):
    """The x6 upper-range children (whose x6^3 aux lb crosses the numeric cap)
    must certify real bounds, not collapse to the 0.0 objective floor.

    Pre-fix: the pinned ``[+inf, +inf)`` aux box made the simplex report
    ``unbounded`` and the floor fallback returned ``optimal 0.0`` — killing the
    x6 branching lever on exactly these children. Post-fix they certify bounds
    above the parent's 57435, monotone in x6, and sound (≤ the true optimum).
    """
    r, lb, ub = config_node
    relaxer = MccormickLPRelaxer(r)
    parent = relaxer.solve_at_node(lb.copy(), ub.copy())
    assert parent.status == "optimal"
    edges = np.linspace(float(lb[6]), float(ub[6]), 5)
    for i in range(4):  # the high-x6 children cross the x6^3 numeric cap
        lo, hi = lb.copy(), ub.copy()
        lo[6], hi[6] = edges[i], edges[i + 1]
        res = relaxer.solve_at_node(lo.copy(), hi.copy())
        assert res.status == "optimal", f"child {i}: {res.status}"
        assert res.lower_bound > float(parent.lower_bound) + 1000.0, (
            f"child {i} bound {res.lower_bound} did not improve on the parent "
            f"{parent.lower_bound} — the 0.0-floor collapse is back"
        )
        assert res.lower_bound <= EX1252_OPT + 1e-2, "bound above the optimum is unsound"


def test_crossed_box_returns_infeasible_not_error(config_node):
    """A genuinely crossed (empty) box — e.g. a binary with lb=1, ub=0, exactly
    what OBBT hands back with ``infeasible=True`` — must return the correct
    ``infeasible`` verdict, not crash the build into ``status="error"``."""
    r, lb, ub = config_node
    lo, hi = lb.copy(), ub.copy()
    lo[36], hi[36] = 1.0, 0.0  # crossed binary: empty box
    res = MccormickLPRelaxer(r).solve_at_node(lo, hi)
    assert res.status == "infeasible", f"expected infeasible, got {res.status}"


def test_hair_crossed_box_is_repaired_not_pruned(config_node):
    """A crossing within float round-off must NOT be declared infeasible — it is
    repaired by widening to the enclosing box and solved normally (a false prune
    here could fathom the true optimum).

    The hair-crossed point box is placed at the parent LP solution's x12 value
    (an LP-feasible point of this box by construction), so the widened box is
    genuinely feasible: the correct outcome is a normal ``optimal`` solve, not an
    empty-box prune and not a build crash.
    """
    r, lb, ub = config_node
    parent = MccormickLPRelaxer(r).solve_at_node(lb.copy(), ub.copy())
    assert parent.status == "optimal" and parent.x is not None
    sol12 = float(parent.x[12])
    lo, hi = lb.copy(), ub.copy()
    lo[12], hi[12] = sol12 + 5e-13, sol12 - 5e-13  # hair-crossed feasible point
    res = MccormickLPRelaxer(r).solve_at_node(lo, hi)
    assert res.status == "optimal", (
        f"round-off crossing must repair-and-solve, got {res.status} "
        "(infeasible here would be a false prune; error a build crash)"
    )
    assert res.lower_bound <= EX1252_OPT + 1e-2
