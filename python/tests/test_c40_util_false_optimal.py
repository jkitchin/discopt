"""Soundness lock (C-40): the incumbent-cutoff FBBT bound-map must never corrupt
the global box and fathom the optimum-containing node.

``util`` (MINLPLib type MBQCP: mixed-binary, four bilinear product constraints
``v3*v0 + v2*v1``, ``v3*v4``, ``v3*v5``, ``v3*v6``; linear objective; MINLPLib
optimum ``=opt= 999.5787502``) was certified ``status="optimal"`` at
``obj = bound = 1072.9614`` in 3 nodes on the DEFAULT (all-flags-OFF) path — a
**false-optimal certificate**: the reported dual (lower) bound 1072.96 exceeds the
true minimum 999.58, an impossible underestimator for a minimize problem.

Root cause (issue C-40) — root-cause class (b), unsound presolve/bound-tightening.
This is the *same failure family* as C-38 (a false-infeasible fathom of the
optimum-containing node) but a **different mechanism**: not the McCormick-LP Farkas
path (C-38, already fixed). Here, after a feasibility-pump incumbent (obj 1072.96)
was found, the per-incumbent cutoff-FBBT phase (``solver.py``: the
``_model_repr.fbbt_with_cutoff`` block) mapped the Rust FBBT result onto the flat
global ``lb``/``ub`` arrays. ``_model_repr`` returned **144** intervals while the
flat B&B has **145** columns (a reformulated/eliminated variable layout); the old
``fbbt_lbs[bi]`` → ``lb[flat_idx]`` map then read a *misaligned* variable's bound,
wrote a **crossed** ``lb > ub`` box in place, and — with the write done in place and
the resulting index-out-of-bounds swallowed — left ~17 entries of the GLOBAL box
corrupted (e.g. ``lb[68]=104.22 > ub[68]=2.2``). The child-node cutoff clamp then
intersected each child box with that corrupted global box, found an *empty*
intersection on the child containing the true optimum (``x[68]=1.2`` lies inside the
child box but outside the corrupted global bound), and fathomed it as "outside the
cutoff box". The surviving nodes' frontier bound collapsed to the incumbent, and the
tree certified 1072.96 as ``optimal``.

The fix. The optional cutoff-FBBT tightening is applied only when the Rust repr's
variable layout provably aligns 1:1 with the flat B&B columns (exactly ``n_vars``
intervals AND every model block scalar), and the intersection is committed only when
it stays consistent (no crossed bound). On a misaligned repr the tightening is
skipped — a valid, looser box, sound by construction (CLAUDE.md §3): never a lost
bound, never a corrupted box, never a fathomed feasible region.

This test is intentionally *unmarked* (CI-visible): the false certificate lands in a
3-node solve, so it reproduces cheaply and the regression must never be invisible to
CI. On ``main`` (pre-fix) the headline assertion fails (dual bound 1072.96 >
999.58); after the fix the dual bound is a valid underestimator (~999.28) and no
false ``optimal`` is certified.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import pytest
from discopt.modeling.core import from_nl

_DATA = Path(__file__).parent / "data" / "minlplib"

# MINLPLib oracle for util.
_UTIL_OPT = 999.5787502  # =opt= (the true global minimum)


# The soundness check runs to the solver's own 60 s time_limit; with JAX compile
# the wall is ~60-65 s, which straddles the 120 s PR-fast --timeout on slower CI
# runners. The per-test marker overrides the CLI cap so this correctness guard
# stays in the PR-fast tier (unmarked / CI-visible) without a false CI timeout.
@pytest.mark.timeout(300)
def test_util_dual_bound_is_valid():
    """The DEFAULT-path dual bound must not exceed the true optimum (no false
    underestimator), and no false ``optimal`` certificate may be issued."""
    r = from_nl(str(_DATA / "util.nl")).solve(time_limit=60, gap_tolerance=1e-4)

    # Headline soundness invariant: for a minimize problem the certified dual
    # (lower) bound is a valid underestimator, so it can never exceed the true
    # optimum. Pre-fix this was 1072.96 > 999.58 — a false certificate.
    assert r.bound is not None, "expected a finite dual bound"
    assert r.bound <= _UTIL_OPT + 1e-4, (
        f"unsound dual bound {r.bound!r} exceeds true optimum {_UTIL_OPT} "
        f"(false underestimator / false-optimal certificate — C-40)"
    )

    # An incumbent, if reported, cannot beat the true optimum.
    if r.objective is not None:
        assert r.objective >= _UTIL_OPT - 1e-4, (
            f"incumbent {r.objective!r} is below the true optimum {_UTIL_OPT}"
        )

    # A ``status="optimal"`` certificate is only honest when the certified value
    # actually agrees with the true optimum to correctness tolerance. Pre-fix the
    # solver certified ``optimal`` at 1072.96 — the false certificate this locks out.
    if r.status == "optimal" and r.objective is not None:
        assert abs(r.objective - _UTIL_OPT) <= 1e-3 * max(1.0, abs(_UTIL_OPT)), (
            f"false-optimal certificate: status=optimal at obj={r.objective!r} "
            f"but true optimum is {_UTIL_OPT}"
        )
