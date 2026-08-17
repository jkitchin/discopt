"""Composition + regression capstone for the lifted-envelope work (issue #154).

Increments 1–3 each added a lifted envelope to the McCormick relaxation:

* **#1** finitized the root FBBT so the lifted aux chains have finite bounds;
* **#2** lifts a sqrt whose argument carries a cross-term monomial
  (``sqrt(x**2 + 2*x*y*z + y**2)``), with a conditioning guard that abstains when
  the folded product-aux bounds reach the magnitude where the fast simplex
  mis-solves the LP;
* **#3** lifts a division with a *non-constant* numerator (``(0.5*x3)/x6``) as a
  factorable reciprocal × numerator product.

Each increment shipped its own adversarial soundness suite. What none of them
locked — and what the *composition* of all three demands — is a single
CI-visible regression that exercises every affected MINLPLib instance across the
two axes where the increments introduced soundness risk:

* **Both LP backends.** The fast Rust ``simplex`` is the default and can return a
  wrong "optimal" objective on an ill-conditioned lifted LP (the #158 hazard);
  the reference ``auto``/HiGHS path does not. A lifted envelope is only sound if
  *both* agree on the soundness direction. The other increments' MINLPLib
  regression cases are ``@pytest.mark.correctness`` — invisible to CI, which runs
  ``-m "not slow and not correctness and ..."``. This file is unmarked, so the
  cross-backend lock actually runs in CI.

* **Default and FBBT-tightened bounds.** FBBT widens/narrows the box the lifts
  are built over; nvs05/nvs22's C4 cross-term reaches ``|g| ~ 1.2e9`` only on the
  FBBT box, which is exactly where the conditioning guard has to fire. Default
  bounds and FBBT bounds therefore stress different code paths.

The invariant is the project's non-negotiable one: a valid lower bound must NEVER
exceed the true optimum. We assert it for every (instance, backend, bound-set)
cell. A non-``optimal`` status (e.g. the fast simplex stopping at
``iteration_limit`` on a wide FBBT box) is acceptable — looseness and early
termination are fine; the only thing forbidden is a *finite* bound above the
optimum. We additionally require that the two backends never disagree on a finite
bound by more than a tolerance (a disagreement is the fingerprint of an
ill-conditioned lifted LP that one backend mis-solved).

These root bounds are honest but mostly loose/inert: none of these five instances
certifies from the root bound alone (the lifted constraints bind aux columns the
objective never reads). The point of this file is not tightening — it is a
permanent, CI-visible soundness floor under the composed lifts so a future
envelope change cannot silently reintroduce a super-optimal dual bound.
"""

import math
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.model_utils import flat_variable_bounds
from discopt.solver import _extract_variable_info
from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

_DATA = Path(__file__).parent / "data" / "minlplib"

# The instances touched by increments 1–3 (lifted sqrt / division / FBBT aux
# chains), with their MINLPLib optima. Each must yield a sound root bound on
# both backends under both bound sets.
_AFFECTED = [
    ("nvs05", 5.47093),
    ("nvs20", 230.922),
    ("nvs22", 6.0584),
    ("ex1252", 128893.8),
    ("chance", 29.8945),
]

_BACKENDS = ["simplex", "auto"]

# No per-solve wall-clock cap is needed. ex1252's FBBT-tightened box used to make
# the fast simplex grind ~54s (its ill-scaled lifted LP broke the basis
# factorization, returning a Numerical bound that the MILP B&B could never fathom,
# so it enumerated the whole tree — issue #170). Equilibration scaling fixed that:
# every cell here now solves to ``optimal`` in well under a second, so the test
# runs the relaxer to completion and exercises the full solve path.


def _bounds_for(m):
    """Return ``(default_bounds, fbbt_bounds)`` as ``(lb, ub)`` pairs."""
    dlb, dub = flat_variable_bounds(m)
    dlb = np.asarray(dlb, dtype=float)
    dub = np.asarray(dub, dtype=float)
    _, elb, eub, io, isz = _extract_variable_info(m)
    tlb, tub, infeasible, _ = tighten_root_bounds_with_fbbt(m, elb.copy(), eub.copy(), io, isz)
    return (dlb, dub), (None if infeasible else (np.asarray(tlb), np.asarray(tub)))


def _root_bound(m, backend, bounds):
    relaxer = MccormickLPRelaxer(m)
    relaxer._backend = backend
    lb, ub = bounds
    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    return relaxer.solve_at_node(lb, ub)


def _root_bound_and_lp(m, backend, bounds):
    """Root bound plus **the LP the separation chain actually ended on**.

    ``solve_at_node`` mutates one ``milp`` object in place through the whole
    separator chain, so shadowing the first separator on the *instance* (not the
    class) is enough to grab a handle on it; the returned object carries every
    row the chain added. Returns ``(result, milp)`` with ``milp is None`` when the
    chain was never entered.
    """
    relaxer = MccormickLPRelaxer(m)
    relaxer._backend = backend
    captured = {}
    inner = relaxer._separate_multilinear

    def _capture(milp, *args, **kwargs):
        captured["milp"] = milp
        return inner(milp, *args, **kwargs)

    relaxer._separate_multilinear = _capture
    lb, ub = bounds
    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    res = relaxer.solve_at_node(lb, ub)
    return res, captured.get("milp")


@pytest.mark.parametrize("instance, optimum", _AFFECTED)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_affected_instance_sound_on_both_backends_and_bound_sets(instance, optimum, backend):
    """For every increment-1–3 instance, the root McCormick bound is sound on this
    backend under *both* the default box and the FBBT-tightened box.

    Soundness only — not optimality and not finiteness. The fast simplex may stop
    at ``iteration_limit`` on a wide FBBT box, and an abstaining lift may drop the
    objective entirely (no finite bound). Both are acceptable. The single
    non-negotiable assertion: a finite bound, if returned, never exceeds the
    optimum."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    default_b, fbbt_b = _bounds_for(m)
    bound_sets = [("default", default_b)]
    if fbbt_b is not None:
        bound_sets.append(("fbbt", fbbt_b))

    for tag, bounds in bound_sets:
        res = _root_bound(m, backend, bounds)
        if res.lower_bound is None or not math.isfinite(res.lower_bound):
            continue
        assert res.lower_bound <= optimum + 1e-3, (
            f"[{instance}/{backend}/{tag}] UNSOUND root bound {res.lower_bound} > optimum {optimum}"
        )


@pytest.mark.parametrize("instance, optimum", _AFFECTED)
def test_affected_instance_backends_agree_on_finite_bound(instance, optimum):
    """The fast simplex must never report a bound **above the true optimum of the
    LP it actually solved**, as re-solved by the reference ``auto``/HiGHS
    backend. That is the #158 wrong-"optimal" hazard: an ill-conditioned lifted LP
    on which the fast backend's answer drifts *upward*, past the value a rigorous
    solve of the same system certifies.

    The fast path reports the Neumaier–Shcherbina *safe* bound (issue #356), a
    rigorous under-estimate of the LP optimum that the reference path reports
    directly, so the two need not be equal: the fast bound may legitimately be
    *lower* (a conservative safe bound), and on free/unbounded lifted columns the
    safe bound's FBBT-bounded box makes it looser still. Only the unsound
    direction — fast above the reference solve of its own LP — is forbidden.

    **Why the comparison captures the LP instead of running both chains** (#1065,
    measured on nvs05/fbbt). This test used to run ``solve_at_node`` once per
    backend and compare the two reported bounds. Those are not two solves of one
    LP: cut separation is driven by ``res.x``, the LP vertex, and the backends
    land on different vertices among alternate optima, so each chain separates a
    different cut set and the comparison is between two different polytopes.
    Instrumenting the separators on nvs05's FBBT box::

        simplex: _separate_univariate_square +6 rows, _separate_convex +40 rows
        auto:    _separate_univariate_square +3 rows, _separate_convex +15 rows

    46 cuts against 18 — the fast chain's bound was higher (2.5477 vs 2.5357)
    because its relaxation was *tighter*, not because it mis-solved anything. On
    the identical 46-row LP the reference certifies 2.5529, comfortably above the
    2.5477 the fast path reported, and both sit under the 5.47093 optimum. The
    divergence is not new machinery, only newly reachable: restoring the affine
    lower bound that ``0 * inf`` was discarding lets ``_separate_convex`` fire on
    these boxes at all, so the round counts stopped coinciding. Capturing the
    chain's own ``milp`` and re-solving it makes both sides the same polytope,
    which is what the hazard is about — and is strictly sharper than the old form,
    since a genuine mis-solve can no longer hide behind a cut-count difference.

    Where a backend returns no finite bound there is nothing to compare; the
    ``<= optimum`` direction is locked by the companion test above."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    default_b, fbbt_b = _bounds_for(m)
    bound_sets = [("default", default_b)]
    if fbbt_b is not None:
        bound_sets.append(("fbbt", fbbt_b))

    n_compared = 0
    for tag, bounds in bound_sets:
        fast, milp = _root_bound_and_lp(m, "simplex", bounds)
        if fast.lower_bound is None or not math.isfinite(fast.lower_bound):
            continue
        assert milp is not None, (
            f"[{instance}/{tag}] the separation chain never ran, so no LP was "
            "captured and this cell compared nothing"
        )
        ref = milp.solve(backend="auto")
        if ref.objective is None or not math.isfinite(ref.objective):
            continue
        n_compared += 1
        # The fast simplex's safe bound must not exceed the reference optimum of
        # the same LP (relative tolerance for large optima like ex1252's ~1.3e5).
        # Being conservatively lower is the safe bound working as designed.
        scale = max(1.0, abs(fast.lower_bound), abs(ref.objective))
        assert fast.lower_bound <= ref.objective + 1e-4 * scale, (
            f"[{instance}/{tag}] fast simplex bound {fast.lower_bound} exceeds the "
            f"reference solve of its own LP ({ref.objective}) — fast bound unreliable"
        )
        assert fast.lower_bound <= optimum + 1e-3 and ref.objective <= optimum + 1e-3, (
            f"[{instance}/{tag}] UNSOUND: simplex={fast.lower_bound} "
            f"auto-on-same-LP={ref.objective} > optimum {optimum}"
        )

    # CLAUDE.md §6: every ``continue`` above is a cell that asserted nothing, so
    # an instance where they all fire would pass vacuously. Measured: each of the
    # five compares on at least one bound set (nvs05 and nvs22 only on ``fbbt``,
    # where the reference reaches ``optimal`` rather than ``iteration_limit``).
    assert n_compared >= 1, (
        f"[{instance}] no (backend, bound-set) cell produced two finite bounds — "
        "the cross-backend lock measured nothing on this instance"
    )
