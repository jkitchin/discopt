"""Reusable soundness audit for RLT cut families.

The correctness-critical property of *every* RLT family is invariant across
levels and constraint types: an RLT cut is a product of non-negative factors
(a constraint slack ``b - a^T x >= 0`` times a bound factor ``x_j - l >= 0`` /
``u - x_j >= 0``), linearized over the lifted product columns. At any genuine
feasible point — where ``X = x x^T`` makes the linearization exact and every
factor is non-negative — the product is non-negative, so the cut can never
remove a feasible point. It may only tighten the relaxation.

This module turns that invariant into a black-box test harness: build the
standard polynomial lifted layout, drive a cut generator at a deliberately
inconsistent moment point so it actually separates, then assert every generated
cut is satisfied at the lifted image of many points sampled from the *true*
feasible region. Any RLT family added later (level-2 products, equality-factor
RLT, nonlinear-constraint-factor RLT) is expected to pass the same audit — call
:func:`audit_bound_factor_cuts` (or :func:`assert_cuts_admit_feasible_points`
with that family's own generator) from its test.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Callable, Optional

import numpy as np
from discopt._jax.cutting_planes import LinearCut
from discopt._jax.rlt_cuts import rlt_constraint_bound_cut, rlt_quadratic_bound_cut_row

Constraint = tuple[dict[int, float], float]  # (a: col -> coeff, b) meaning a^T x <= b
Bound = tuple[float, float]
# Quadratic factor ``g(x) {<=,==} 0`` with
#   g = const + sum_i lin_i x_i + sum_{(k,l)} quad_{kl} x_k x_l
# Keys of ``quad`` are sorted index pairs ``(k, l)`` with ``k <= l``.
QuadForm = tuple[dict[tuple[int, int], float], dict[int, float], float, str]


def standard_lifted_info(n: int) -> tuple[dict, int]:
    """Standard polynomial lifted layout over ``n`` original variables.

    Columns ``0..n-1`` are the originals; then every square ``x_i^2`` and every
    pairwise product ``x_i x_j`` (i < j) gets its own lifted column. This is the
    column map (``info``) that the RLT cut functions address.

    Returns ``(info, n_total)``.
    """
    original = {i: i for i in range(n)}
    monomial: dict[tuple[int, int], int] = {}
    bilinear: dict[tuple[int, int], int] = {}
    col = n
    for i in range(n):
        monomial[(i, 2)] = col
        col += 1
    for i in range(n):
        for j in range(i + 1, n):
            bilinear[(i, j)] = col
            col += 1
    info = {"original": original, "monomial": monomial, "bilinear": bilinear}
    return info, col


def lift(info: dict, x: np.ndarray, n_total: int) -> np.ndarray:
    """Lift an original-variable point ``x`` into the full column space ``z``.

    ``z`` places ``x`` in the original columns and the *exact* products
    ``x_i x_j`` / ``x_i^2`` in their lifted columns — i.e. the genuine moment
    image ``X = x x^T`` of a feasible point.
    """
    z = np.zeros(n_total, dtype=np.float64)
    for col, idx in info.get("original", {}).items():
        z[idx] = x[col]
    for (i, _p), idx in info.get("monomial", {}).items():
        z[idx] = x[i] * x[i]
    for (i, j), idx in info.get("bilinear", {}).items():
        z[idx] = x[i] * x[j]
    return z


def sample_feasible(
    bounds: list[Bound],
    constraints: list[Constraint],
    *,
    count: int,
    seed: int = 0,
    max_tries_factor: int = 200,
) -> list[np.ndarray]:
    """Rejection-sample ``count`` points satisfying box ``bounds`` and every
    ``a^T x <= b`` in ``constraints``. Raises if the region is too thin to fill.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    out: list[np.ndarray] = []
    tries = 0
    budget = count * max_tries_factor
    while len(out) < count and tries < budget:
        tries += 1
        x = rng.uniform(lo, hi)
        if all(sum(a.get(i, 0.0) * x[i] for i in a) <= b + 1e-12 for a, b in constraints):
            out.append(x)
    if len(out) < count:
        raise RuntimeError(
            f"feasible region too thin: got {len(out)}/{count} samples in {tries} tries"
        )
    return out


def _inconsistent_moment_point(info: dict, bounds: list[Bound], n_total: int) -> np.ndarray:
    """A point with originals at the box midpoint but *inflated* product columns
    (``X != x x^T``), so RLT generators actually separate a cut to audit."""
    z = np.zeros(n_total, dtype=np.float64)
    mid = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=np.float64)
    for col, idx in info.get("original", {}).items():
        z[idx] = mid[col]
    for (i, _p), idx in info.get("monomial", {}).items():
        lo, hi = bounds[i]
        z[idx] = max(lo * lo, hi * hi) + 1.0  # above the secant — violates
    for (i, j), idx in info.get("bilinear", {}).items():
        (li, hii), (lj, hij) = bounds[i], bounds[j]
        z[idx] = max(li * lj, li * hij, hii * lj, hii * hij) + 1.0
    return z


def generate_bound_factor_cuts(
    constraints: list[Constraint],
    bounds: list[Bound],
    info: dict,
    n_total: int,
    *,
    moment_point: Optional[np.ndarray] = None,
) -> list[LinearCut]:
    """Every level-1 constraint×bound-factor RLT cut separated at ``moment_point``
    (an inflated, inconsistent moment point by default)."""
    if moment_point is None:
        moment_point = _inconsistent_moment_point(info, bounds, n_total)
    cuts: list[LinearCut] = []
    for a, b in constraints:
        for j in range(len(bounds)):
            lo, hi = bounds[j]
            for lower, bnd in ((True, lo), (False, hi)):
                if not np.isfinite(bnd):
                    continue
                cut = rlt_constraint_bound_cut(
                    a, b, j, float(bnd), lower, info, moment_point, n_total
                )
                if cut is not None:
                    cuts.append(cut)
    return cuts


def assert_cuts_admit_feasible_points(
    cuts: list[LinearCut],
    info: dict,
    n_total: int,
    samples: list[np.ndarray],
    *,
    tol: float = 1e-9,
) -> None:
    """Assert no cut removes any sampled feasible point (the soundness invariant).

    Each cut is ``coeffs . z >= rhs``; at the lifted image of a feasible point
    the slack must be ``>= -tol``.
    """
    worst = np.inf
    for cut in cuts:
        for x in samples:
            z = lift(info, x, n_total)
            slack = float(cut.coeffs @ z - cut.rhs)
            worst = min(worst, slack)
    assert worst >= -tol, f"an RLT cut removed a feasible point (worst slack {worst:.3e})"


def audit_bound_factor_cuts(
    constraints: list[Constraint],
    bounds: list[Bound],
    *,
    n_samples: int = 3000,
    seed: int = 0,
    tol: float = 1e-9,
    cut_generator: Optional[Callable[..., list[LinearCut]]] = None,
) -> int:
    """End-to-end soundness audit for an RLT cut family. Returns the cut count.

    Builds the standard lifted layout for the variables referenced by
    ``bounds``, generates cuts (default: level-1 bound-factor RLT, or any
    ``cut_generator(constraints, bounds, info, n_total)`` for a new family),
    samples feasible points, and asserts none is cut. Also asserts at least one
    cut was actually separated, so the audit is never vacuously green.
    """
    n = len(bounds)
    info, n_total = standard_lifted_info(n)
    gen = cut_generator or generate_bound_factor_cuts
    cuts = gen(constraints, bounds, info, n_total)
    assert cuts, "no RLT cut was separated — audit would be vacuous"
    samples = sample_feasible(bounds, constraints, count=n_samples, seed=seed)
    assert_cuts_admit_feasible_points(cuts, info, n_total, samples, tol=tol)
    return len(cuts)


# ── Phase 2: nonlinear (quadratic) constraint-factor RLT audit ──────────────
#
# Degree-3 lifted layout: original columns, then a column for every product
# monomial of total degree 2 or 3 over ``n`` variables (squares, cubes, pairs,
# ``x_i**2 x_j`` mixed terms, and distinct triples). A uniform ``product`` map
# keyed by the *sorted multiset* of indices addresses them, so a single resolver
# serves the quadratic-factor cut assembler for every monomial it can emit.


def standard_lifted_info_quadratic(n: int) -> tuple[dict, int]:
    """Lifted layout carrying every degree-2 and degree-3 monomial over ``n`` vars.

    Returns ``(info, n_total)`` where ``info["original"]`` maps each variable to
    its column and ``info["product"]`` maps each sorted multiset of 2 or 3 indices
    to its lifted column.
    """
    original = {i: i for i in range(n)}
    product: dict[tuple[int, ...], int] = {}
    col = n
    for degree in (2, 3):
        for combo in combinations_with_replacement(range(n), degree):
            product[combo] = col
            col += 1
    info = {"original": original, "product": product}
    return info, col


def lift_quadratic(info: dict, x: np.ndarray, n_total: int) -> np.ndarray:
    """Lift ``x`` into the degree-3 product space (exact monomial values)."""
    z = np.zeros(n_total, dtype=np.float64)
    for idx, col in info.get("original", {}).items():
        z[col] = x[idx]
    for key, col in info.get("product", {}).items():
        val = 1.0
        for i in key:
            val *= x[i]
        z[col] = val
    return z


def _quadratic_resolvers(info: dict) -> tuple[Callable, Callable]:
    orig = info.get("original", {})
    prod = info.get("product", {})

    def orig_col(i: int) -> Optional[int]:
        return orig.get(i)

    def prod_col(key: tuple[int, ...]) -> Optional[int]:
        return prod.get(tuple(sorted(key)))

    return orig_col, prod_col


def generate_quadratic_bound_factor_cuts(
    quad_forms: list[QuadForm],
    bounds: list[Bound],
    info: dict,
    n_total: int,
    *,
    moment_point: Optional[np.ndarray] = None,
) -> list[LinearCut]:
    """Every quadratic-factor × bound-factor RLT row, as ``coeffs . z >= rhs``
    (``>=`` for ``<=`` parents, ``==`` for equality parents). ``moment_point`` is
    unused (rows are generated unconditionally, not separated) and accepted only
    for signature parity with :func:`generate_bound_factor_cuts`."""
    del moment_point
    orig_col, prod_col = _quadratic_resolvers(info)
    cuts: list[LinearCut] = []
    for quad, lin, const, sense in quad_forms:
        for j in range(len(bounds)):
            lo, hi = bounds[j]
            for lower, bnd in ((True, lo), (False, hi)):
                if not np.isfinite(bnd):
                    continue
                row = rlt_quadratic_bound_cut_row(
                    quad, lin, const, j, float(bnd), lower, orig_col, prod_col, n_total
                )
                if row is None:
                    continue
                coeffs, rhs = row
                cut_sense = "==" if sense == "==" else ">="
                cuts.append(LinearCut(coeffs=coeffs, rhs=float(rhs), sense=cut_sense))
    return cuts


def assert_quadratic_cuts_admit_feasible_points(
    cuts: list[LinearCut],
    info: dict,
    n_total: int,
    samples: list[np.ndarray],
    *,
    tol: float = 1e-9,
) -> None:
    """Assert no quadratic-factor RLT row removes a feasible point. ``>=`` rows
    must have non-negative slack; ``==`` rows must hold to ``tol`` (the product of
    a zero equality factor is identically zero at any feasible point)."""
    worst_ineq = np.inf
    worst_eq = 0.0
    for cut in cuts:
        for x in samples:
            z = lift_quadratic(info, x, n_total)
            resid = float(cut.coeffs @ z - cut.rhs)
            if cut.sense == "==":
                worst_eq = max(worst_eq, abs(resid))
            else:
                worst_ineq = min(worst_ineq, resid)
    assert worst_ineq >= -tol, (
        f"a quadratic-factor RLT cut removed a feasible point (worst slack {worst_ineq:.3e})"
    )
    assert worst_eq <= tol, (
        f"a quadratic-factor equality RLT row was violated at a feasible point "
        f"(worst |residual| {worst_eq:.3e})"
    )


def sample_feasible_quadratic(
    bounds: list[Bound],
    quad_forms: list[QuadForm],
    *,
    count: int,
    seed: int = 0,
    max_tries_factor: int = 400,
    tol: float = 1e-9,
) -> list[np.ndarray]:
    """Rejection-sample ``count`` points satisfying box ``bounds`` and every
    *inequality* quadratic form ``g(x) <= 0``. Equality forms are ignored for
    sampling (their surface is measure zero); audit equality rows with points
    constructed on the surface instead."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    def _g(quad, lin, const, x):
        val = const + sum(c * x[i] for i, c in lin.items())
        val += sum(c * x[ka] * x[kb] for (ka, kb), c in quad.items())
        return val

    out: list[np.ndarray] = []
    tries = 0
    budget = count * max_tries_factor
    while len(out) < count and tries < budget:
        tries += 1
        x = rng.uniform(lo, hi)
        if all(
            _g(quad, lin, const, x) <= tol
            for quad, lin, const, sense in quad_forms
            if sense == "<="
        ):
            out.append(x)
    if len(out) < count:
        raise RuntimeError(
            f"feasible region too thin: got {len(out)}/{count} samples in {tries} tries"
        )
    return out


def audit_quadratic_bound_factor_cuts(
    quad_forms: list[QuadForm],
    bounds: list[Bound],
    *,
    n_samples: int = 3000,
    seed: int = 0,
    tol: float = 1e-7,
) -> int:
    """End-to-end soundness audit for the quadratic constraint-factor RLT family.

    Generates every quadratic-factor × bound-factor row over a full degree-3
    lifted layout, rejection-samples points feasible for the inequality forms,
    and asserts no ``>=`` row cuts one off (and every ``==`` row holds exactly).
    Returns the row count. Asserts at least one row was generated.
    """
    n = len(bounds)
    info, n_total = standard_lifted_info_quadratic(n)
    cuts = generate_quadratic_bound_factor_cuts(quad_forms, bounds, info, n_total)
    assert cuts, "no quadratic-factor RLT row was generated — audit would be vacuous"
    samples = sample_feasible_quadratic(bounds, quad_forms, count=n_samples, seed=seed)
    assert_quadratic_cuts_admit_feasible_points(cuts, info, n_total, samples, tol=tol)
    return len(cuts)
