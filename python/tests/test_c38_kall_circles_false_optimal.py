"""Soundness lock (C-38): a McCormick-LP node ``infeasible`` verdict must never
fathom without a *verified Farkas certificate*.

``kall_circles_c8a`` (circle packing; reverse-convex non-overlap constraints
``(x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2``; MINLPLib optimum
``=best= 2.5409191380``, ``=bestdual= 2.5409129340``) was certified
``status="optimal"`` at ``obj = bound = 3.6142`` in 3 nodes on the DEFAULT
(all-flags-OFF) path — a **false-optimal certificate**: the reported dual (lower)
bound 3.6142 exceeds the true minimum 2.5409, an impossible underestimator.

Root cause (issue C-38). The in-house warm-started dual simplex returns a
*numerical false* ``infeasible`` on the ill-conditioned lifted McCormick relaxation
of this class (coefficient spread only ~1e2–1e4 — well under any conditioning
heuristic; HiGHS proves every one of these LPs feasible), and it does so with **no
verified Farkas ray** — both cold and after equilibration. The per-node LP path
trusted that uncertified verdict (``mccormick_lp.py``: the cold-path
``if res.status == "infeasible": return infeasible`` and the incremental
``_reverify``'s ``if not ill: return infeasible`` / equilibrated-infeasible trust),
fathoming the sub-box that contains the true optimum. The surviving nodes' bounds
then exceed 2.5409, and the tree certifies a false optimum.

The fix. A node fathom on ``infeasible`` is rigorous ONLY when backed by a verified
Farkas dual ray (the sole independent proof of LP emptiness). An uncertified
``infeasible`` from the fragile simplex is treated as inconclusive (no fathom): the
node stays open on its inherited (valid) parent bound and branches. This forgoes a
*possible* prune, never a valid bound — sound by construction.

This test is intentionally *unmarked* (CI-visible): the false certificate lands in
a 3-node solve, so it reproduces cheaply and the regression must never be invisible
to CI. On ``main`` (pre-fix) the headline assertion fails (dual bound 3.6142 >
2.5409); after the fix the dual bound is a valid underestimator and no false
``optimal`` is certified.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

from discopt.modeling.core import from_nl

_DATA = Path(__file__).parent / "data" / "minlplib"

# MINLPLib oracle for kall_circles_c8a.
_KALL_C8A_BEST = 2.5409191380  # =best= (a feasible objective; the true optimum)
_KALL_C8A_BESTDUAL = 2.5409129340  # =bestdual=


def test_kall_circles_c8a_dual_bound_is_valid():
    """The DEFAULT-path dual bound must not exceed the true optimum (no false
    underestimator), and no false ``optimal`` certificate may be issued."""
    r = from_nl(str(_DATA / "kall_circles_c8a.nl")).solve(time_limit=25, gap_tolerance=1e-4)

    # Headline soundness invariant: for a minimize problem the certified dual
    # (lower) bound is a valid underestimator, so it can never exceed a feasible
    # objective. Pre-fix this was 3.6142 > 2.5409 — a false certificate.
    assert r.bound is not None, "expected a finite dual bound"
    assert r.bound <= _KALL_C8A_BEST + 1e-4, (
        f"unsound dual bound {r.bound!r} exceeds true optimum {_KALL_C8A_BEST} "
        f"(false underestimator / false-optimal certificate — C-38)"
    )

    # An incumbent, if reported, cannot beat the true optimum.
    if r.objective is not None:
        assert r.objective >= _KALL_C8A_BEST - 1e-4, (
            f"incumbent {r.objective!r} is below the true optimum {_KALL_C8A_BEST}"
        )

    # A ``status="optimal"`` certificate is only honest when the certified value
    # actually agrees with the true optimum to correctness tolerance. Pre-fix the
    # solver certified ``optimal`` at 3.6142 — the false certificate this locks out.
    if r.status == "optimal" and r.objective is not None:
        assert abs(r.objective - _KALL_C8A_BEST) <= 1e-3 * max(1.0, abs(_KALL_C8A_BEST)), (
            f"false-optimal certificate: status=optimal at obj={r.objective!r} "
            f"but true optimum is {_KALL_C8A_BEST}"
        )
