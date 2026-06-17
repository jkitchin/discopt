"""Theorem-style property harness for relaxation rules.

Implements the testing discipline from the SOTA rule inventory: *every relaxation
rule should be tested as a theorem*. A relaxation ``(cv, cc)`` for a factorable
expression ``f`` over a box ``[lb, ub]`` must satisfy, for the proof to be valid:

1. **Containment** — every true graph point satisfies ``cv(x) <= f(x) <= cc(x)``.
   A single violated point invalidates the global optimality certificate.
2. **Corner exactness** — at the box vertices the envelope is tight
   (``cv == cc == f``) for the McCormick-exact primitives (bilinear, monotone
   convex/concave, powers). Tested opt-in since wide-interval trig falls back to
   range bounds that are not corner-exact.
3. **Monotone tightening** — as the box shrinks around a point the envelope gap
   ``cc - cv`` is non-increasing and tends to zero. This is *why* spatial
   branching converges.
4. **Convexity classification** — no false convex/concave verdicts from the DCP
   detector (a wrong verdict can produce an invalid convex-handler relaxation).

These helpers are shared by ``test_relaxation_theorems.py`` (parametrized over
the full operator-coverage matrix) and are reusable by any future relaxation work.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.relaxation_compiler import compile_relaxation
from discopt.modeling.core import Model

# Tolerances mirror conftest numerical policy (abs=1e-6); we use a slightly
# looser absolute slack to absorb float64 round-off in the vmapped evaluators.
CONTAINMENT_TOL = 1e-7
EXACTNESS_TOL = 1e-6
TIGHTENING_TOL = 1e-7

Bounds = Sequence[tuple[float, float]]


def build_relaxation(
    expr_fn: Callable,
    bounds: Bounds,
) -> tuple[Callable, Callable, jnp.ndarray, jnp.ndarray]:
    """Compile a relaxation + true evaluator for ``expr_fn`` over ``bounds``.

    ``expr_fn`` receives one variable handle per entry in ``bounds`` and returns
    a single scalar expression. Returns ``(relax_fn, true_fn, lb, ub)`` where
    ``relax_fn(x_cv, x_cc, lb, ub) -> (cv, cc)`` and ``true_fn(x) -> value`` use
    the flat variable layout.
    """
    m = Model("harness")
    handles = [m.continuous(f"x{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(bounds)]
    expr = expr_fn(*handles)
    # A minimize target is required for a well-formed model; the objective itself
    # is irrelevant to relaxation compilation of ``expr``.
    m.minimize(handles[0])

    relax_fn = compile_relaxation(expr, m)
    true_fn = compile_expression(expr, m)

    lb = jnp.array([lo for lo, _ in bounds], dtype=jnp.float64)
    ub = jnp.array([hi for _, hi in bounds], dtype=jnp.float64)
    return relax_fn, true_fn, lb, ub


def _sample_points(
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    n_interior: int,
    seed: int,
) -> jnp.ndarray:
    """Interior (random) + boundary-face + corner samples within ``[lb, ub]``."""
    rng = np.random.default_rng(seed)
    n = lb.shape[0]
    width = np.maximum(np.asarray(ub) - np.asarray(lb), 1e-12)

    # Interior: strictly inside to avoid coinciding with corners.
    t = rng.uniform(0.02, 0.98, size=(n_interior, n))
    interior = np.asarray(lb) + t * width

    # Corners: all 2^n vertices (capped for high dimension).
    if n <= 8:
        corners = np.array(list(itertools.product(*zip(np.asarray(lb), np.asarray(ub)))))
    else:
        corners = np.stack([np.asarray(lb), np.asarray(ub)])

    # Boundary faces: pin one coordinate to a bound, randomize the rest.
    faces = []
    for j in range(n):
        for bound in (np.asarray(lb)[j], np.asarray(ub)[j]):
            pt = np.asarray(lb) + rng.uniform(0.02, 0.98, size=n) * width
            pt[j] = bound
            faces.append(pt)
    faces = np.array(faces) if faces else np.empty((0, n))

    pts = np.concatenate([interior, corners, faces], axis=0)
    return jnp.array(pts, dtype=jnp.float64)


def evaluate(
    relax_fn: Callable,
    true_fn: Callable,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    xs: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vmapped evaluation of ``(cv, cc, f)`` over a batch of points ``xs``."""

    @jax.jit
    def _eval(xs):
        cv_cc = jax.vmap(lambda x: relax_fn(x, x, lb, ub))(xs)
        f = jax.vmap(true_fn)(xs)
        return cv_cc, f

    (cv, cc), f = _eval(xs)
    return np.asarray(cv), np.asarray(cc), np.asarray(f)


def assert_containment(
    expr_fn: Callable,
    bounds: Bounds,
    *,
    n_interior: int = 2000,
    seed: int = 0,
    tol: float = CONTAINMENT_TOL,
    label: Optional[str] = None,
) -> None:
    """Theorem 1: ``cv(x) <= f(x) <= cc(x)`` for all sampled true graph points."""
    relax_fn, true_fn, lb, ub = build_relaxation(expr_fn, bounds)
    xs = _sample_points(lb, ub, n_interior, seed)
    cv, cc, f = evaluate(relax_fn, true_fn, lb, ub, xs)

    cv_viol = int(np.sum(cv > f + tol))
    cc_viol = int(np.sum(cc < f - tol))
    name = label or getattr(expr_fn, "__name__", "expr")
    if cv_viol or cc_viol:
        worst_cv = float(np.max(cv - f)) if cv.size else 0.0
        worst_cc = float(np.max(f - cc)) if cc.size else 0.0
        raise AssertionError(
            f"[{name}] containment violated over {bounds}: "
            f"cv>f at {cv_viol} pts (worst +{worst_cv:.2e}), "
            f"cc<f at {cc_viol} pts (worst +{worst_cc:.2e})"
        )


def assert_corner_exactness(
    expr_fn: Callable,
    bounds: Bounds,
    *,
    tol: float = EXACTNESS_TOL,
    label: Optional[str] = None,
) -> None:
    """Theorem 2: envelope is tight (cv == cc == f) at every box vertex.

    Only valid for McCormick-exact primitives (bilinear, monotone convex/concave,
    powers). Wide-interval periodic relaxations are intentionally *not* corner
    exact and should not be checked here.
    """
    relax_fn, true_fn, lb, ub = build_relaxation(expr_fn, bounds)
    corners = jnp.array(
        list(itertools.product(*zip(np.asarray(lb), np.asarray(ub)))),
        dtype=jnp.float64,
    )
    cv, cc, f = evaluate(relax_fn, true_fn, lb, ub, corners)
    name = label or getattr(expr_fn, "__name__", "expr")
    gap_cv = float(np.max(np.abs(cv - f))) if cv.size else 0.0
    gap_cc = float(np.max(np.abs(cc - f))) if cc.size else 0.0
    if gap_cv > tol or gap_cc > tol:
        raise AssertionError(
            f"[{name}] corner exactness failed over {bounds}: "
            f"max|cv-f|={gap_cv:.2e}, max|cc-f|={gap_cc:.2e}"
        )


def assert_monotone_tightening(
    expr_fn: Callable,
    bounds: Bounds,
    *,
    shrink_factors: Sequence[float] = (1.0, 0.5, 0.25, 0.1, 0.02),
    tol: float = TIGHTENING_TOL,
    label: Optional[str] = None,
) -> None:
    """Theorem 3: the envelope gap ``cc - cv`` is non-increasing as the box
    shrinks symmetrically toward its center, and tends to zero.

    The compiled ``relax_fn`` already takes ``(lb, ub)`` as arguments, so we
    evaluate the same relaxation on progressively tighter boxes without
    recompiling, measuring the gap at the (fixed) center point.
    """
    relax_fn, true_fn, lb0, ub0 = build_relaxation(expr_fn, bounds)
    center = 0.5 * (lb0 + ub0)
    half = 0.5 * (ub0 - lb0)

    name = label or getattr(expr_fn, "__name__", "expr")
    prev_gap = np.inf
    first_gap = None
    last_gap = np.inf
    for s in shrink_factors:
        lb = center - s * half
        ub = center + s * half
        cv, cc = relax_fn(center, center, lb, ub)
        gap = float(np.asarray(cc) - np.asarray(cv))
        gap = max(gap, 0.0)
        if gap > prev_gap + tol:
            raise AssertionError(
                f"[{name}] gap increased as box shrank over {bounds}: "
                f"{prev_gap:.3e} -> {gap:.3e} at factor {s}"
            )
        if first_gap is None:
            first_gap = gap
        prev_gap = gap
        last_gap = gap
    # The tightest box must collapse the gap toward zero. The gap on a smooth
    # node shrinks with the box (e.g. ~width^2 for a quadratic), so the
    # vanishing criterion is relative to the initial gap rather than absolute.
    assert first_gap is not None
    if last_gap > max(1e-6, 0.05 * first_gap):
        raise AssertionError(
            f"[{name}] gap did not collapse as box shrank over {bounds}: "
            f"{first_gap:.3e} -> {last_gap:.3e}"
        )


def assert_convexity(
    expr_fn: Callable,
    bounds: Bounds,
    expected: str,
    *,
    label: Optional[str] = None,
) -> None:
    """Theorem 4: the DCP detector returns ``expected`` and never a false
    convex/concave verdict.

    ``expected`` is one of ``"CONVEX"``, ``"CONCAVE"``, ``"AFFINE"`` or
    ``"UNKNOWN"``. For ``"UNKNOWN"`` we only require that the detector does not
    *over*-claim convex or concave (UNKNOWN/AFFINE are acceptable; a wrong
    CONVEX/CONCAVE is a soundness failure).
    """
    from discopt._jax.convexity import Curvature, classify_expr

    m = Model("harness")
    handles = [m.continuous(f"x{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(bounds)]
    expr = expr_fn(*handles)
    m.minimize(handles[0])
    verdict = classify_expr(expr, m)

    name = label or getattr(expr_fn, "__name__", "expr")
    if expected == "UNKNOWN":
        # Only flag an over-claim (a definite wrong convex/concave label).
        if verdict in (Curvature.CONVEX, Curvature.CONCAVE):
            raise AssertionError(
                f"[{name}] DCP over-claimed {verdict.name} for a non-convex/concave expr"
            )
        return
    want = Curvature[expected]
    # AFFINE is a strict refinement of both CONVEX and CONCAVE; accept it.
    if (
        verdict == want
        or verdict == Curvature.AFFINE
        and want
        in (
            Curvature.CONVEX,
            Curvature.CONCAVE,
        )
    ):
        return
    raise AssertionError(f"[{name}] DCP verdict {verdict.name}, expected {expected}")
