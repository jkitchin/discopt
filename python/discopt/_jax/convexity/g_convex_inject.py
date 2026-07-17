"""Flag-gated injection of rigorous G-convexity transformation cuts (#181).

The graduation slice of the convex-transformable program: recognize
constraints whose body is **G-convex** (a strictly larger class than
DCP-convex; see :mod:`g_convexity`) and inject *rigorously valid* linear cuts
that expose the transformed convex shape to the LP node relaxation — the
mechanism KMS 2012 §3/§7 describe, wired into discopt's existing structure-cut
presolve alongside the GP / signed-signomial injectors.

**Default OFF.** Gated on ``DISCOPT_G_CONVEX_CUTS`` (a bound-changing change,
CLAUDE.md §5). This module never runs unless the flag is set; graduation to
default-ON requires the flag-ON-vs-OFF differential panel (cert-clean +
net-positive).

Construction (rigorous — removes no feasible point)
---------------------------------------------------
A constraint is normalized to ``body(x) ≤ 0``. If ``body`` is G-convex on the
variable box with witness ``ρ > 0`` (``ρ = 0`` is ordinary convexity, already
handled by the existing OA machinery — skipped here), then ``h(x) =
exp(ρ·body(x))`` is **convex** on the box and, because ``exp`` is increasing,

    ``body(x) ≤ 0  ⟺  h(x) ≤ exp(ρ·0) = 1``.

For a linearization point ``x₀`` take the float gradient ``g = ∇h(x₀)`` (only a
*direction* — its accuracy affects strength, never validity). The intercept is
made rigorous with sound interval arithmetic:

    ``c = lower bound of  (h(x) − g·x)  over the box``   (via ``evaluate_interval``),

so ``c ≤ h(x) − g·x`` for **every** ``x`` in the box. Hence for any
original-feasible ``x`` (``h(x) ≤ 1``):

    ``g·x = (g·x + c) − c ≤ h(x) − c ≤ 1 − c``,

and the injected linear cut ``g·x ≤ 1 − c`` (RHS rounded outward) is satisfied
by every feasible point — it only tightens the relaxation, it never removes a
feasible ``x``. This is the same "no feasible x removed" contract the GP
injector proves, but the intercept here is an *interval* lower bound rather
than an exact affine tangent, which keeps it rigorous even though ``body`` is a
general nonlinear (merely G-convex) body.

The ``≥`` case is the mirror: ``body ≥ 0 ⟺ −body ≤ 0``; when ``−body`` is
G-convex (i.e. ``body`` is G-concave) the same construction applies to
``−body``.
"""

from __future__ import annotations

import os

import numpy as np

from discopt.modeling.core import Constraint, Model, Variable

from . import interval as iv
from .g_convexity import certify_g_convex
from .interval import Interval, _round_down, _round_up
from .interval_ad import interval_hessian
from .interval_eval import evaluate_interval

# Cap on injected cuts and linearization points, keeping presolve cheap.
_MAX_CONSTRAINTS = 32
_DEFAULT_POINTS = 3


def g_convex_cuts_enabled() -> bool:
    """Whether ``DISCOPT_G_CONVEX_CUTS`` enables the injector (default OFF)."""
    return os.environ.get("DISCOPT_G_CONVEX_CUTS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _scalar_offsets(model: Model) -> "dict[int, Variable] | None":
    """Map each flat index to its (scalar) variable, or ``None`` if any
    variable is non-scalar.

    The first graduation slice targets scalar-variable bodies so the
    flat-gradient → model-expression mapping is unambiguous. A model with any
    array variable is skipped (returns ``None``) — sound: simply no cut.
    """
    offsets: dict[int, Variable] = {}
    off = 0
    for v in model._variables:
        if v.size != 1:
            return None
        offsets[off] = v
        off += 1
    return offsets


def _declared_box(model: Model) -> dict:
    box: dict = {}
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=float).ravel()
        ub = np.asarray(v.ub, dtype=float).ravel()
        if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
            # Unbounded variable — the box certificate would abstain anyway;
            # record it so the caller can bail on this constraint.
            box[v] = Interval(lb, ub)
        else:
            box[v] = Interval(lb, ub)
    return box


def _linear_expr(coeffs: np.ndarray, offsets: "dict[int, Variable]"):
    """Build the model expression ``Σ_j coeffs[j] · v_j`` over scalar vars."""
    expr = None
    for j, v in offsets.items():
        cj = float(coeffs[j])
        if cj == 0.0:
            continue
        term = cj * v
        expr = term if expr is None else expr + term
    return expr


def inject_g_convex_cuts(
    model: Model,
    *,
    points: int = _DEFAULT_POINTS,
    max_constraints: int = _MAX_CONSTRAINTS,
) -> int:
    """Inject rigorous G-convexity transformation cuts; return the count.

    For each ``≤`` (resp. ``≥``) constraint whose body is certified G-convex
    (resp. G-concave) with witness ``ρ > 0`` on the declared box, adds one or
    more valid linear cuts ``g·x ≤ 1 − c`` (see module docstring) that tighten
    the LP relaxation without removing any feasible point. Bodies that are
    already convex (``ρ = 0``), reference array variables, or whose transformed
    residual has an unbounded interval enclosure are skipped.

    This mutates ``model`` in place (via ``subject_to``) and is a no-op unless
    :func:`g_convex_cuts_enabled`. Any per-constraint failure is swallowed so
    the injector can never break a solve.
    """
    import discopt.modeling as dm

    offsets = _scalar_offsets(model)
    if offsets is None:
        return 0
    box = _declared_box(model)
    # Bail entirely if any variable is unbounded (no finite box to certify on).
    for v in model._variables:
        if not (np.all(np.isfinite(box[v].lo)) and np.all(np.isfinite(box[v].hi))):
            return 0

    lb = np.array([float(np.asarray(box[v].lo).ravel()[0]) for v in model._variables], dtype=float)
    ub = np.array([float(np.asarray(box[v].hi).ravel()[0]) for v in model._variables], dtype=float)
    n = lb.shape[0]

    applied = 0
    # Snapshot the constraint list — we append while iterating.
    for c in list(model._constraints):
        if applied >= max_constraints:
            break
        if not isinstance(c, Constraint) or c.sense not in ("<=", ">="):
            continue
        # Orient so the working body ``phi`` satisfies ``phi(x) ≤ 0`` and we
        # look for G-convexity of ``phi``.
        phi = c.body if c.sense == "<=" else -c.body
        want = "g_convex"
        try:
            cert = certify_g_convex(phi, model, box=box)
        except Exception:
            cert = None
        if cert is None or cert.kind != want:
            continue
        rho = float(cert.rho)
        if not (rho > 0.0):
            # Ordinary convex body — the existing OA path already handles it.
            continue

        try:
            n_new = _inject_for_body(model, phi, rho, box, offsets, lb, ub, n, points, applied, dm)
        except Exception:
            n_new = 0
        applied += n_new
    return applied


def rigorous_g_convex_cut_coeffs(model, phi, rho, x0, box, *, offsets=None):
    """Rigorously valid transformation cut ``g·x ≤ rhs`` for ``phi(x) ≤ 0``.

    Given a body ``phi`` certified G-convex on ``box`` with witness ``rho > 0``
    and a linearization point ``x0`` (flat, over the model's scalar variables),
    returns ``(g, rhs)`` — a length-``n`` gradient over the flat variable vector
    and a scalar RHS — such that **every** ``x`` in ``box`` with ``phi(x) ≤ 0``
    satisfies ``g·x ≤ rhs``. Returns ``None`` when the enclosure is unusable
    (unsupported atom, non-finite, degenerate gradient).

    The construction is the module's rigorous tangent: ``h = exp(rho·phi)`` is
    convex on ``box`` (the g_convex witness), so with ``g`` the interval-midpoint
    of ``∇h(x0)`` the intercept ``c = ψ_lo(x0) − Σ_i hw_i·r_i`` (``ψ = h − g·x``,
    ``hw`` the gradient-interval half-widths, ``r`` the box half-widths) is a
    sound lower bound of ``min_box ψ``; hence ``g·x ≤ h(x) − c ≤ 1 − c = rhs``
    for feasible ``x``. Shared by the model-constraint injector and the per-node
    relaxer separator.
    """
    import discopt.modeling as dm

    variables = list(model._variables)
    x0 = np.asarray(x0, dtype=float).ravel()
    lb = np.array([float(np.asarray(box[v].lo).ravel()[0]) for v in variables], dtype=float)
    ub = np.array([float(np.asarray(box[v].hi).ravel()[0]) for v in variables], dtype=float)
    r = 0.5 * (ub - lb)

    pbox = {v: Interval(np.array([x0[i]]), np.array([x0[i]])) for i, v in enumerate(variables)}
    try:
        ad0 = interval_hessian(phi, model, box=pbox)
    except ValueError:
        return None
    phi_iv = ad0.value
    grad_iv = ad0.grad
    if not (
        np.all(np.isfinite(phi_iv.lo))
        and np.all(np.isfinite(phi_iv.hi))
        and np.all(np.isfinite(grad_iv.lo))
        and np.all(np.isfinite(grad_iv.hi))
    ):
        return None

    # ∇h(x0) enclosure and its midpoint direction g; h certified convex on box.
    h_iv = iv.exp(Interval.point(rho) * phi_iv)
    dh_iv = Interval.point(rho) * h_iv * grad_iv
    g = np.asarray(dh_iv.mid, dtype=float).ravel()
    hw = 0.5 * (
        np.asarray(dh_iv.hi, dtype=float).ravel() - np.asarray(dh_iv.lo, dtype=float).ravel()
    )
    if not np.all(np.isfinite(g)) or np.allclose(g, 0.0):
        return None

    if offsets is None:
        offsets = {i: v for i, v in enumerate(variables)}
    lin = _linear_expr(g, offsets)
    residual = dm.exp(rho * phi) if lin is None else (dm.exp(rho * phi) - lin)
    try:
        psi0_iv = evaluate_interval(residual, model, box=pbox)
    except Exception:
        return None
    psi0_lo = float(np.asarray(psi0_iv.lo).ravel()[0])
    if not np.isfinite(psi0_lo):
        return None
    correction = float(_round_up(np.float64(np.sum(hw * r))))
    c = float(_round_down(np.float64(psi0_lo - correction)))
    rhs = float(_round_up(np.float64(1.0 - c)))
    if not np.isfinite(rhs):
        return None
    return g, rhs


def _inject_for_body(model, phi, rho, box, offsets, lb, ub, n, points, tag, dm) -> int:
    """Emit the valid linear cuts for one G-convex body ``phi`` (``phi ≤ 0``)."""
    # Linearization points: midpoint + per-variable high corners (like the GP
    # injector), each yields an independent valid cut.
    mid = 0.5 * (lb + ub)
    pts = [mid]
    for j in range(n):
        if len(pts) >= points:
            break
        p = mid.copy()
        p[j] = ub[j]
        pts.append(p)

    emitted = 0
    for p_idx, x0 in enumerate(pts[:points]):
        coeffs = rigorous_g_convex_cut_coeffs(model, phi, rho, x0, box, offsets=offsets)
        if coeffs is None:
            continue
        g, rhs = coeffs
        cut_lin = _linear_expr(g, offsets)
        if cut_lin is None:
            continue
        model.subject_to(cut_lin <= rhs, name=f"gconv_cut_{tag}_{p_idx}")
        emitted += 1
    return emitted


__all__ = [
    "g_convex_cuts_enabled",
    "inject_g_convex_cuts",
    "rigorous_g_convex_cut_coeffs",
]
