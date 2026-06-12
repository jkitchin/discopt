"""Geometric programming: detection, log-space reformulation, and solve.

A *geometric program* (GP) in standard form is

    minimize    f_0(x)
    subject to  f_i(x) <= 1,    i = 1..m   (posynomial inequalities)
                g_j(x) == 1,    j = 1..p   (monomial equalities)
                x > 0

where ``f_i`` are posynomials and ``g_j`` are monomials (see
:mod:`discopt._jax.convexity.posynomial`). A posynomial is not convex in
``x``, but under ``y_j = log(x_j)`` every monomial
``c * prod_j x_j^{a_j}`` becomes ``exp(a^T y + log c)``; a posynomial
becomes a sum of exponentials of affine forms (convex), a monomial
equality becomes an affine equality, and a posynomial inequality
``f_i <= 1`` becomes ``sum_k exp(affine_k) <= 1`` (a convex sublevel
set). The transformed program is therefore a convex NLP in ``y``.

This module:

* :func:`classify_gp` — recognise GP structure in a :class:`Model`.
* :func:`as_geometric_program` — build the convex log-space NLP plus the
  ``x = exp(y)`` recovery map. Returns ``None`` for non-GP models.
* :func:`solve_gp` — solve via the log-space convex formulation and map
  the solution back to ``x``-space.

The surface is additive: it does not touch the soundness contract of the
DCP walker (posynomials remain ``UNKNOWN`` there). References: Boyd &
Vandenberghe Ch. 4.5; Agrawal et al. 2019 (Disciplined GP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import discopt.modeling as dm
from discopt._jax.convexity.posynomial import (
    Monomial,
    PosynomialForm,
    is_posynomial,
)
from discopt.modeling.core import (
    Constant,
    Constraint,
    Expression,
    Model,
    ObjectiveSense,
    SolveResult,
    VarType,
)

_TOL = 1e-12

# An affine form over the log-variables: (coeff_by_offset, constant).
Affine = tuple[dict[int, float], float]


# ──────────────────────────────────────────────────────────────────────
# GP structure (the classification result, still in x-space terms)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GPConstraint:
    """A single GP constraint in normalised ``posynomial vs monomial`` form.

    For an inequality the meaning is ``lhs <= rhs`` where ``lhs`` is a
    posynomial and ``rhs`` is a single monomial; for an equality both
    sides are monomials (``lhs == rhs``).
    """

    lhs: PosynomialForm
    rhs: Monomial
    is_equality: bool
    source: Optional[Constraint] = None


@dataclass
class GPStructure:
    """Recognised GP structure of a model (objective + constraints)."""

    objective: PosynomialForm
    minimize: bool
    constraints: list[GPConstraint]


# ──────────────────────────────────────────────────────────────────────
# Body splitting
# ──────────────────────────────────────────────────────────────────────


def _split_signed_monomials(
    body: Expression, model: Model
) -> Optional[tuple[list[Monomial], list[Monomial]]]:
    """Split a constraint body into (positive_monomials, negative_monomials).

    Returns ``None`` when any term of the flattened body is not a
    monomial on the strictly-positive box. The two lists carry
    positive-coefficient monomials; a term with overall negative sign
    lands in the second list with its coefficient negated to positive.
    """
    from discopt._jax.convexity.posynomial import _flatten_sum_terms, _parse_monomial

    terms: list[tuple[float, Expression]] = []
    _flatten_sum_terms(body, 1.0, terms)

    plus: list[Monomial] = []
    minus: list[Monomial] = []
    for scale, term in terms:
        mono = _parse_monomial(term, model)
        if mono is None:
            return None
        coeff = mono.coeff * scale
        if abs(coeff) <= _TOL:
            continue
        if coeff > 0:
            plus.append(Monomial(coeff, mono.exponents))
        else:
            minus.append(Monomial(-coeff, mono.exponents))
    return plus, minus


def _classify_constraint(constraint: Constraint, model: Model) -> Optional[GPConstraint]:
    """Classify one constraint as a GP constraint, or return ``None``.

    Inequalities are stored as ``body <= 0`` (``>=`` is normalised to
    ``<=`` upstream). ``body = plus - minus`` then means
    ``plus_posynomial <= minus``; this is a GP inequality iff ``minus`` is
    a single monomial. An ``==`` constraint is a GP equality iff both
    sides are single monomials.
    """
    if constraint.sense not in ("<=", "=="):
        return None
    split = _split_signed_monomials(constraint.body, model)
    if split is None:
        return None
    plus, minus = split

    if constraint.sense == "==":
        if len(plus) != 1 or len(minus) != 1:
            return None
        return GPConstraint(
            lhs=PosynomialForm([plus[0]]),
            rhs=minus[0],
            is_equality=True,
            source=constraint,
        )

    # Inequality: plus_posynomial <= single monomial.
    if len(plus) < 1 or len(minus) != 1:
        return None
    return GPConstraint(
        lhs=PosynomialForm(plus),
        rhs=minus[0],
        is_equality=False,
        source=constraint,
    )


# ──────────────────────────────────────────────────────────────────────
# Model-level classification
# ──────────────────────────────────────────────────────────────────────


def _all_variables_positive_continuous(model: Model) -> bool:
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            return False
        if float(np.asarray(v.lb).min()) <= 0.0:
            return False
    return True


def is_log_convex(model: Model) -> bool:
    """Return ``True`` iff ``model`` is a geometric program.

    A GP is convex under the change of variables ``y = log x`` (every
    monomial becomes ``exp`` of an affine form, a posynomial a convex
    sum-of-exponentials). This is a verdict in **log-space**, kept
    deliberately separate from
    :func:`discopt._jax.convexity.classify_model`, which reports
    convexity in the **original** ``x``-space.

    The distinction matters: a genuine GP is generally *not* convex in
    ``x`` (a posynomial Hessian is indefinite on the positive orthant),
    so for such a model ``classify_model(model, use_certificate=True)``
    returns ``(False, ...)`` while ``is_log_convex(model)`` returns
    ``True``. Folding log-convexity into the ``x``-space verdict would
    mis-gate the ``x``-space convex fast path — a soundness break — so the
    two are exposed as independent predicates.

    This is whole-model recognition (the model is a GP in standard form);
    per-expression log-curvature lattice propagation is a future
    extension. A ``True`` result is a proof: it is exactly the precondition
    under which :func:`as_geometric_program` builds an equivalent convex
    NLP in ``y``.
    """
    return classify_gp(model) is not None


def classify_gp(model: Model) -> Optional[GPStructure]:
    """Return the :class:`GPStructure` of ``model`` if it is a GP, else ``None``.

    Requirements: every variable is continuous with a strictly positive
    lower bound; the objective is a posynomial minimisation (or a
    monomial maximisation); every constraint is a posynomial ``<=``
    monomial inequality or a monomial ``==`` monomial equality.
    """
    if model._objective is None:
        return None
    if not _all_variables_positive_continuous(model):
        return None
    # Reject models carrying non-algebraic constraint kinds (indicator,
    # disjunctive, SOS, logical) — only plain Constraint objects are GP.
    if any(not isinstance(c, Constraint) for c in model._constraints):
        return None

    obj_expr = model._objective.expression
    obj_form = is_posynomial(obj_expr, model)
    minimize = model._objective.sense == ObjectiveSense.MINIMIZE
    if obj_form is None:
        return None
    if not minimize and not obj_form.is_monomial:
        # Maximising a multi-term posynomial is not GP-representable.
        return None

    gp_constraints: list[GPConstraint] = []
    for constraint in model._constraints:
        gp_c = _classify_constraint(constraint, model)
        if gp_c is None:
            return None
        gp_constraints.append(gp_c)

    return GPStructure(objective=obj_form, minimize=minimize, constraints=gp_constraints)


# ──────────────────────────────────────────────────────────────────────
# Log-space reformulation
# ──────────────────────────────────────────────────────────────────────


def _total_scalars(model: Model) -> int:
    return sum(v.size for v in model._variables)


def _log_bounds(model: Model, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-scalar-offset (log lb, log ub) arrays; ub == inf maps to inf."""
    log_lb = np.full(n, -np.inf, dtype=np.float64)
    log_ub = np.full(n, np.inf, dtype=np.float64)
    offset = 0
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        for k in range(v.size):
            log_lb[offset + k] = math.log(lb[k])
            log_ub[offset + k] = math.log(ub[k]) if np.isfinite(ub[k]) else np.inf
        offset += v.size
    return log_lb, log_ub


def _mono_affine(mono: Monomial) -> Affine:
    """Monomial ``c * prod x^a`` -> affine ``a^T y + log c`` in log-space."""
    return dict(mono.exponents), math.log(mono.coeff)


def _affine_minus(a: Affine, b: Affine) -> Affine:
    """Affine difference ``a - b`` (log of a monomial ratio)."""
    coeffs = dict(a[0])
    for off, c in b[0].items():
        coeffs[off] = coeffs.get(off, 0.0) - c
    return coeffs, a[1] - b[1]


def _affine_expr(affine: Affine, y) -> Expression:
    """Build a discopt expression ``sum_off coeff * y[off] + const``."""
    coeffs, const = affine
    expr: Expression = Constant(np.asarray(const, dtype=np.float64))
    for off in sorted(coeffs):
        c = coeffs[off]
        if abs(c) > _TOL:
            expr = expr + c * y[off]
    return expr


def _sum_exp_expr(affines: list[Affine], y) -> Expression:
    """Build ``sum_k exp(affine_k)`` (a convex posynomial in log-space)."""
    expr: Optional[Expression] = None
    for affine in affines:
        term = dm.exp(_affine_expr(affine, y))
        expr = term if expr is None else expr + term
    assert expr is not None
    return expr


@dataclass
class GeometricProgram:
    """A model recognised as a GP, with its convex log-space formulation.

    Attributes
    ----------
    original : Model
        The source model (in ``x``-space).
    structure : GPStructure
        The recognised posynomial/monomial structure.
    log_model : Model
        The convex NLP in ``y = log(x)``.
    n_scalars : int
        Number of flat scalar variables (length of ``y``).
    """

    original: Model
    structure: GPStructure
    log_model: Model
    n_scalars: int

    def recover_x(self, y_values: np.ndarray) -> dict[str, np.ndarray]:
        """Map a log-space solution ``y`` back to ``x = exp(y)`` per variable."""
        x_flat = np.exp(np.asarray(y_values, dtype=np.float64))
        out: dict[str, np.ndarray] = {}
        offset = 0
        for v in self.original._variables:
            chunk = x_flat[offset : offset + v.size]
            out[v.name] = chunk.reshape(v.shape) if v.shape else chunk.reshape(())
            offset += v.size
        return out

    def objective_value(self, y_values: np.ndarray) -> float:
        """Evaluate the original posynomial objective at ``x = exp(y)``."""
        return _posynomial_value(self.structure.objective, np.asarray(y_values))


def _posynomial_value(form: PosynomialForm, y: np.ndarray) -> float:
    total = 0.0
    for mono in form.monomials:
        exponent = math.log(mono.coeff)
        for off, a in mono.exponents.items():
            exponent += a * float(y[off])
        total += math.exp(exponent)
    return total


def as_geometric_program(model: Model) -> Optional[GeometricProgram]:
    """Build the convex log-space NLP for ``model`` if it is a GP, else ``None``."""
    structure = classify_gp(model)
    if structure is None:
        return None

    n = _total_scalars(model)
    log_lb, log_ub = _log_bounds(model, n)

    log_model = Model(f"{model.name}_log")
    y = log_model.continuous("y", shape=(n,), lb=log_lb, ub=log_ub)

    # Objective.
    obj = structure.objective
    obj_affines = [_mono_affine(mono) for mono in obj.monomials]
    if obj.is_monomial:
        # min/max of a monomial == min/max of its (affine) log.
        affine_expr = _affine_expr(obj_affines[0], y)
        if structure.minimize:
            log_model.minimize(affine_expr)
        else:
            log_model.maximize(affine_expr)
    else:
        # Posynomial minimisation: minimise sum_k exp(affine_k) (convex).
        log_model.minimize(_sum_exp_expr(obj_affines, y))

    # Constraints.
    for gp_c in structure.constraints:
        rhs_affine = _mono_affine(gp_c.rhs)
        lhs_affines = [_mono_affine(mono) for mono in gp_c.lhs.monomials]
        # Divide through by the rhs monomial: each lhs term becomes a
        # monomial ratio (affine in log-space).
        ratios = [_affine_minus(a, rhs_affine) for a in lhs_affines]

        if gp_c.is_equality:
            # monomial == monomial  ->  affine == 0.
            log_model.subject_to(_affine_expr(ratios[0], y) == 0.0)
        elif len(ratios) == 1:
            # monomial <= monomial  ->  affine <= 0 (linear).
            log_model.subject_to(_affine_expr(ratios[0], y) <= 0.0)
        else:
            # posynomial <= monomial  ->  sum_k exp(affine_k) <= 1.
            log_model.subject_to(_sum_exp_expr(ratios, y) <= 1.0)

    return GeometricProgram(
        original=model,
        structure=structure,
        log_model=log_model,
        n_scalars=n,
    )


# ──────────────────────────────────────────────────────────────────────
# Solve
# ──────────────────────────────────────────────────────────────────────


def solve_gp(model: Model, **solve_kwargs) -> Optional[SolveResult]:
    """Solve ``model`` as a GP via its log-space convex formulation.

    Returns a :class:`SolveResult` in the original ``x``-space, or
    ``None`` if ``model`` is not a geometric program.
    """
    gp = as_geometric_program(model)
    if gp is None:
        return None

    log_result = gp.log_model.solve(**solve_kwargs)
    if not isinstance(log_result, SolveResult):
        # Streaming solve (callback/iterator) is not supported for GP.
        raise TypeError("solve_gp does not support streaming solve options")

    result = SolveResult(status=log_result.status)
    result.wall_time = log_result.wall_time
    result.bound = None
    result.gap = log_result.gap

    if log_result.x is not None and "y" in log_result.x:
        y_values = np.asarray(log_result.x["y"], dtype=np.float64).reshape(-1)
        result.x = gp.recover_x(y_values)
        result.objective = gp.objective_value(y_values)

    # The log-space program is convex and equivalent to the original GP, so an
    # ``optimal`` log-space solution is the *global* optimum of the GP. Its
    # objective is therefore simultaneously the best incumbent and a valid lower
    # bound (a zero-gap, single-NLP solve). Only assert this when the convex
    # solve actually converged — on a limit/infeasible status we must not
    # fabricate a bound (correctness invariant). Guarded on a finite objective
    # so a NaN/inf recovery never poses as a certified optimum.
    if (
        log_result.status == "optimal"
        and result.objective is not None
        and math.isfinite(result.objective)
    ):
        result.bound = result.objective
        result.gap = 0.0
        result.convex_fast_path = True
    return result


__all__ = [
    "GPConstraint",
    "GPStructure",
    "GeometricProgram",
    "as_geometric_program",
    "classify_gp",
    "is_log_convex",
    "solve_gp",
]
