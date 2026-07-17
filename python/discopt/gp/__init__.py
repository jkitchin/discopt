"""Geometric programming: detection, log-space reformulation, and solve.

A *geometric program* (GP) in standard form is::

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

import heapq
import itertools
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import discopt.modeling as dm
from discopt._jax.convexity.log_lattice import (
    LogCurvature,
    classify_log_curvature,
    log_combine_product,
    log_combine_sum,
    log_negate,
)
from discopt._jax.convexity.posynomial import (
    Monomial,
    PosynomialForm,
    is_posynomial,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    Model,
    ObjectiveSense,
    SolveResult,
    UnaryOp,
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

    This is whole-model recognition (the model is a GP in standard form).
    For the finer-grained, *per-expression* verdict — a log-curvature
    lattice (``LOG_AFFINE``/``LOG_CONVEX``/``LOG_CONCAVE``) propagated
    through the DAG so a sub-expression deep inside a constraint can be
    tagged and composed upward — see :func:`classify_log_curvature`. A
    ``True`` result here is a proof: it is exactly the precondition under
    which :func:`as_geometric_program` builds an equivalent convex NLP in
    ``y``.
    """
    return classify_gp(model) is not None


def _classify_gp_body(model: Model) -> Optional[GPStructure]:
    """Recognise the posynomial/monomial structure, sans variable-type gate.

    Shared by :func:`classify_gp` (which additionally requires every
    variable to be continuous) and :func:`classify_gp_minlp` (which admits
    strictly-positive integer variables). Both callers must have already
    checked :meth:`Model._has_builder_only_rows` and the strict-positivity
    of every variable's lower bound before calling this — those gates are
    the soundness preconditions of the posynomial recogniser, and this body
    does not re-check them.
    """
    # Callers guarantee a non-None objective (both check it before delegating);
    # assert it so the type narrows and the contract is explicit.
    assert model._objective is not None
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


def classify_gp(model: Model) -> Optional[GPStructure]:
    """Return the :class:`GPStructure` of ``model`` if it is a GP, else ``None``.

    Requirements: every variable is continuous with a strictly positive
    lower bound; the objective is a posynomial minimisation (or a
    monomial maximisation); every constraint is a posynomial ``<=``
    monomial inequality or a monomial ``==`` monomial equality.
    """
    if model._objective is None:
        return None
    # X-1: fast-API (`m.constraint(...)` linear fast path / `add_linear_constraints`)
    # rows and `add_linear_objective`/`add_quadratic_objective` live only in the Rust
    # builder, invisible to the posynomial recogniser below. Recognising the model as
    # a GP here would solve the log-space reformulation WITHOUT those rows/objective —
    # a wrong certificate on the default solve path (GP-1). A general linear row is
    # GP-representable only in special sign patterns, so refuse (return None) and let
    # the model fall back to spatial B&B, which sees the builder rows.
    if model._has_builder_only_rows():
        return None
    if not _all_variables_positive_continuous(model):
        return None

    return _classify_gp_body(model)


# ──────────────────────────────────────────────────────────────────────
# Per-row log-curvature report (consumer of the per-expression lattice)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ObjectiveLogCurvature:
    """Log-curvature of the objective and whether it is convex in ``y``.

    ``curvature`` is the :class:`LogCurvature` of the objective expression.
    ``log_convex`` is ``True`` iff *optimising* it is a convex problem in
    ``y = log x``: minimising a log-convex (or log-affine) objective, or
    maximising a log-concave (or log-affine) one.
    """

    curvature: LogCurvature
    minimize: bool
    log_convex: bool


@dataclass
class ConstraintLogCurvature:
    """Log-curvature of one constraint's two sides and its feasible set.

    A constraint is normalised to ``body sense 0`` with ``body = lhs -
    rhs``; ``lhs`` / ``rhs`` here are the :class:`LogCurvature` of the
    positive / negative additive parts of ``body`` (each strictly positive
    when not ``UNKNOWN``). ``log_convex`` is ``True`` iff the constraint's
    feasible region is convex in ``y = log x`` under the sound rule in
    :func:`model_log_curvature`.
    """

    source: Constraint
    lhs: LogCurvature
    rhs: LogCurvature
    log_convex: bool


@dataclass
class ModelLogCurvature:
    """Per-row log-curvature report for a model.

    The whole-model :func:`is_log_convex` returns a single yes/no; this
    report instead tags **each row** (objective + every constraint) via the
    per-expression lattice, so a model that is convex-in-``y`` on some rows
    and not others is legible — the *partially* log-convex case that
    motivated issue #115.
    """

    objective: Optional[ObjectiveLogCurvature]
    constraints: list[ConstraintLogCurvature]

    @property
    def is_log_convex_program(self) -> bool:
        """True iff every row is convex in ``y`` (a full GP).

        This is a per-row derivation of the same verdict as
        :func:`is_log_convex`, and is *more* general: it certifies rows
        built from nested posynomials / reciprocals (e.g.
        ``sqrt(x + y) <= t``) that the monomial-splitting
        :func:`classify_gp` rejects.
        """
        obj_ok = self.objective is None or self.objective.log_convex
        return obj_ok and all(c.log_convex for c in self.constraints)

    @property
    def log_convex_rows(self) -> int:
        """Number of constraint rows proven convex in ``y``."""
        return sum(1 for c in self.constraints if c.log_convex)


def _split_additive(
    expr: Expression, sign: float, plus: list[Expression], minus: list[Expression]
) -> None:
    """Group ``expr`` into positive / negative parts at its *top-level* sum.

    Walks only ``+`` / ``-`` / unary ``neg`` so sub-terms such as
    ``sqrt(x + y)`` or ``1 / (x + y)`` stay intact for the log-curvature
    walker to classify. ``sign`` tracks the running coefficient sign.
    """
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _split_additive(expr.left, sign, plus, minus)
        _split_additive(expr.right, sign, plus, minus)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _split_additive(expr.left, sign, plus, minus)
        _split_additive(expr.right, -sign, plus, minus)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _split_additive(expr.operand, -sign, plus, minus)
        return
    (plus if sign > 0 else minus).append(expr)


def _side_log_curvature(terms: list[Expression], cache: dict) -> LogCurvature:
    """Fold the log-curvature of a sum of terms via :func:`log_combine_sum`.

    Returns ``UNKNOWN`` for an empty side (no positive quantity to reason
    about). A positive-coefficient scalar factor inside a term is handled
    by the walker itself (``c * f`` is ``log_combine_product`` of an affine
    coefficient), so no coefficient bookkeeping is needed here.
    """
    result: Optional[LogCurvature] = None
    for term in terms:
        label = classify_log_curvature(term, None, cache)
        result = label if result is None else log_combine_sum(result, label)
        if result == LogCurvature.UNKNOWN:
            return LogCurvature.UNKNOWN
    return result if result is not None else LogCurvature.UNKNOWN


def _constraint_log_curvature(constraint: Constraint, cache: dict) -> ConstraintLogCurvature:
    """Classify a single constraint's feasible-set log-curvature.

    With ``body = P - N`` (positive part ``P``, negative part ``N``) the
    constraint means ``P <= N`` (``<=``), ``P >= N`` (``>=``), or ``P == N``
    (``==``). Writing the ratio ``r = P / N`` (``log_combine_product`` of
    ``P`` against the reciprocal log-curvature of ``N``):

    * ``P <= N`` ⇔ ``log(P/N) <= 0`` is convex iff ``r`` is
      log-convex/affine (a convex sublevel set);
    * ``P >= N`` ⇔ ``log(N/P) <= 0`` is convex iff ``r`` is
      log-concave/affine;
    * ``P == N`` is an affine equality in ``y`` (convex) iff ``r`` is
      log-affine — i.e. monomial == monomial.

    A missing side or an ``UNKNOWN`` side yields ``log_convex=False`` (no
    proof), never a false positive.
    """
    plus: list[Expression] = []
    minus: list[Expression] = []
    _split_additive(constraint.body, 1.0, plus, minus)

    p = _side_log_curvature(plus, cache)
    n = _side_log_curvature(minus, cache)

    log_convex = False
    if p != LogCurvature.UNKNOWN and n != LogCurvature.UNKNOWN:
        ratio = log_combine_product(p, log_negate(n))
        if constraint.sense == "==":
            log_convex = ratio == LogCurvature.LOG_AFFINE
        elif constraint.sense == "<=":
            log_convex = ratio in (LogCurvature.LOG_AFFINE, LogCurvature.LOG_CONVEX)
        elif constraint.sense == ">=":
            log_convex = ratio in (LogCurvature.LOG_AFFINE, LogCurvature.LOG_CONCAVE)

    return ConstraintLogCurvature(source=constraint, lhs=p, rhs=n, log_convex=log_convex)


def model_log_curvature(model: Model) -> ModelLogCurvature:
    """Report the per-row log-curvature of ``model`` (issue #115 consumer).

    Runs the per-expression :func:`classify_log_curvature` lattice over the
    objective and every constraint and derives, for each row, whether it is
    convex in ``y = log x``. Unlike the whole-model :func:`is_log_convex`
    (a single verdict) this exposes *which* rows are log-convex, making a
    **partially** log-convex model legible.

    Soundness: every ``log_convex=True`` is a proof (each side's label
    certifies strict positivity and its log-curvature, and the per-sense
    rule is a sound convex-representability test); a failed precondition
    reports ``log_convex=False`` / ``UNKNOWN``, never a false verdict. This
    is a pure structural analysis — it does not solve, reformulate, or
    touch the x-space convex fast path.

    Only plain algebraic :class:`Constraint` rows are classifiable;
    non-algebraic rows (indicator, disjunctive, SOS, logical) are reported
    ``UNKNOWN`` / not-log-convex rather than skipped.
    """
    cache: dict = {}

    objective: Optional[ObjectiveLogCurvature] = None
    if model._objective is not None:
        minimize = model._objective.sense == ObjectiveSense.MINIMIZE
        curv = classify_log_curvature(model._objective.expression, None, cache)
        if minimize:
            obj_ok = curv in (LogCurvature.LOG_AFFINE, LogCurvature.LOG_CONVEX)
        else:
            obj_ok = curv in (LogCurvature.LOG_AFFINE, LogCurvature.LOG_CONCAVE)
        objective = ObjectiveLogCurvature(curvature=curv, minimize=minimize, log_convex=obj_ok)

    constraints: list[ConstraintLogCurvature] = []
    for constraint in model._constraints:
        if not isinstance(constraint, Constraint):
            constraints.append(
                ConstraintLogCurvature(
                    source=constraint,
                    lhs=LogCurvature.UNKNOWN,
                    rhs=LogCurvature.UNKNOWN,
                    log_convex=False,
                )
            )
            continue
        constraints.append(_constraint_log_curvature(constraint, cache))

    return ModelLogCurvature(objective=objective, constraints=constraints)


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


def _build_log_model(
    structure: GPStructure,
    log_lb: np.ndarray,
    log_ub: np.ndarray,
    name: str,
) -> Model:
    """Build the convex log-space NLP for ``structure`` on the given log-box.

    ``log_lb`` / ``log_ub`` are the per-scalar-offset ``y = log x`` bounds
    (already ``log``-transformed; ``+inf`` upper bounds pass through). The
    same ``structure`` can therefore be re-emitted on a *tightened* box
    per branch-and-bound node without re-running recognition — the log-space
    objective/constraint algebra is identical, only the ``y``-box changes.
    """
    n = int(np.asarray(log_lb).shape[0])
    log_model = Model(name)
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

    return log_model


def as_geometric_program(model: Model) -> Optional[GeometricProgram]:
    """Build the convex log-space NLP for ``model`` if it is a GP, else ``None``."""
    structure = classify_gp(model)
    if structure is None:
        return None

    n = _total_scalars(model)
    log_lb, log_ub = _log_bounds(model, n)
    log_model = _build_log_model(structure, log_lb, log_ub, f"{model.name}_log")

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

    x_values: Optional[dict[str, np.ndarray]] = None
    objective: Optional[float] = None
    if log_result.x is not None and "y" in log_result.x:
        y_values = np.asarray(log_result.x["y"], dtype=np.float64).reshape(-1)
        x_values = gp.recover_x(y_values)
        objective = gp.objective_value(y_values)

    # The log-space program is convex and *exactly* equivalent to the original
    # GP: ``classify_gp`` accepts only an exact posynomial/monomial transform
    # (no approximation), and under ``y = log x`` a posynomial becomes a convex
    # sum-of-exponentials. Hence a converged (``optimal``) log-space solution is
    # a *certified global optimum* of the GP — its recovered objective is
    # simultaneously the best incumbent and a valid global lower bound, so the
    # gap is exactly zero and the result is genuinely certified.
    #
    # This certification is claimed ONLY when all of the following hold, so we
    # never over-claim on a non-optimal / degenerate case:
    #   (a) the model classified as a GP — guaranteed here, since ``gp`` is
    #       non-None, i.e. an *exact* transform, not an approximation;
    #   (b) the convex solve returned status ``"optimal"`` (a certified-optimal
    #       convex solve, not a limit/infeasible/error termination);
    #   (c) a finite objective was recovered (a NaN/inf recovery certifies
    #       nothing and must never pose as an optimum).
    # On any other status we leave ``bound=None`` and ``gap=None`` (an
    # under-claim, which stays sound) rather than fabricate a bound in the wrong
    # (log-space) units.
    certified = (
        log_result.status == "optimal" and objective is not None and math.isfinite(objective)
    )
    if certified:
        bound: Optional[float] = objective
        gap: Optional[float] = 0.0
    else:
        bound = None
        gap = None

    # Construct the result in one shot with its final fields so
    # ``SolveResult.__post_init__`` validates the *actual* bound. Building it
    # with a transient ``bound=None`` and patching afterwards would let
    # ``__post_init__`` silently downgrade ``gap_certified`` to ``False`` (the
    # GP-2 bug). A certified GP optimum thus correctly keeps
    # ``gap_certified=True``.
    result = SolveResult(
        status=log_result.status,
        objective=objective,
        bound=bound,
        gap=gap,
        x=x_values,
        node_count=0,
        wall_time=log_result.wall_time,
        convex_fast_path=certified,
        gap_certified=certified,
        _model=model,
    )
    return result


# ──────────────────────────────────────────────────────────────────────
# GP-MINLP: y-space node relaxations + integer branch-and-bound (#116)
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GPMinlpStructure:
    """A MINLP whose continuous relaxation is a geometric program.

    Attributes
    ----------
    structure : GPStructure
        The recognised posynomial/monomial structure over *all* variables
        (continuous and discrete alike — a discrete variable enters a
        monomial exactly as a positive continuous one does).
    integer_offsets : list of int
        Flat scalar offsets of the discrete (integer) variable components.
        These are the offsets branched on; the continuous block stays in
        ``y = log x``.
    """

    structure: GPStructure
    integer_offsets: list[int]


def _discrete_scalar_offsets(model: Model) -> list[int]:
    """Flat scalar offsets of every discrete (integer/binary) variable component."""
    offsets: list[int] = []
    pos = 0
    for v in model._variables:
        if v.var_type != VarType.CONTINUOUS:
            offsets.extend(range(pos, pos + v.size))
        pos += v.size
    return offsets


def _x_bounds(model: Model, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-scalar-offset ``(lb, ub)`` arrays in ``x``-space (``ub`` may be ``inf``)."""
    x_lb = np.empty(n, dtype=np.float64)
    x_ub = np.empty(n, dtype=np.float64)
    offset = 0
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        x_lb[offset : offset + v.size] = lb
        x_ub[offset : offset + v.size] = ub
        offset += v.size
    return x_lb, x_ub


def classify_gp_minlp(model: Model) -> Optional[GPMinlpStructure]:
    """Recognise a MINLP whose continuous relaxation is a GP, else ``None``.

    Preconditions (any failure → ``None``, i.e. fall back to spatial B&B):

    * The model carries at least one discrete (integer) variable — a pure
      continuous GP is the job of :func:`classify_gp` / :func:`solve_gp`.
    * Every variable — continuous **and** discrete — has a strictly positive
      lower bound. This is the ``lᵢ > 0`` precondition the log map
      ``yᵢ = log xᵢ`` needs at every node (issue #116). Binary ``{0, 1}``
      variables (``lb = 0``) therefore do not qualify here.
    * Every discrete variable has a finite upper bound, so integer
      branch-and-bound terminates.
    * With the discrete variables *relaxed to continuous*, the model is a GP
      in standard form (posynomial objective / ``posynomial ≤ monomial`` /
      ``monomial == monomial`` rows) — recognised by :func:`_classify_gp_body`.

    Soundness: recognition is exact (no approximation). Under these
    preconditions each node's continuous relaxation is a genuine geometric
    program, so its log-space convex optimum is a *rigorous* bound on the
    node — the guarantee :func:`solve_gp_minlp` relies on to certify.
    """
    if model._objective is None:
        return None
    # X-1: builder-only linear rows/objective are invisible to the posynomial
    # recogniser; recognising here would drop them from the reformulation.
    if model._has_builder_only_rows():
        return None

    has_discrete = False
    for v in model._variables:
        if float(np.asarray(v.lb).min()) <= 0.0:
            return None
        if v.var_type != VarType.CONTINUOUS:
            has_discrete = True
            if not np.all(np.isfinite(np.asarray(v.ub, dtype=np.float64))):
                # An unbounded integer variable would make B&B non-terminating.
                return None
    if not has_discrete:
        return None

    structure = _classify_gp_body(model)
    if structure is None:
        return None

    return GPMinlpStructure(
        structure=structure,
        integer_offsets=_discrete_scalar_offsets(model),
    )


def is_gp_minlp(model: Model) -> bool:
    """Return ``True`` iff ``model`` is a GP-structured MINLP (see :func:`classify_gp_minlp`)."""
    return classify_gp_minlp(model) is not None


def _x_flat_to_dict(model: Model, x_flat: np.ndarray) -> dict[str, np.ndarray]:
    """Scatter a flat ``x`` vector back to a per-variable dict (original shapes)."""
    out: dict[str, np.ndarray] = {}
    offset = 0
    for v in model._variables:
        chunk = np.asarray(x_flat[offset : offset + v.size], dtype=np.float64)
        out[v.name] = chunk.reshape(v.shape) if v.shape else chunk.reshape(())
        offset += v.size
    return out


# A node relaxation objective within this of the incumbent is not worth
# branching on: an equally-good subtree is already accounted for by the
# incumbent. Kept far below the integrality/feasibility tolerances so the
# search still finds the *true* optimum (it only prunes genuine ties).
_GP_MINLP_PRUNE_TOL = 1e-9


def _solve_gp_node(
    structure: GPStructure,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    node_index: int,
    solve_kwargs: dict,
) -> tuple[str, Optional[float], Optional[np.ndarray]]:
    """Solve one node's convex log-space relaxation on the box ``[x_lb, x_ub]``.

    Returns ``(status, x_objective, x_flat)``. ``x_objective`` is the original
    ``x``-space posynomial objective value at the relaxation optimum (a valid
    node bound); both it and ``x_flat`` are ``None`` when the convex solve did
    not converge to ``optimal``/``feasible``.
    """
    log_lb = np.log(x_lb)
    log_ub = np.where(np.isfinite(x_ub), np.log(np.where(np.isfinite(x_ub), x_ub, 1.0)), np.inf)
    log_model = _build_log_model(structure, log_lb, log_ub, f"gp_minlp_node{node_index}")
    result = log_model.solve(**solve_kwargs)
    if not isinstance(result, SolveResult):
        raise TypeError("solve_gp_minlp does not support streaming solve options")
    if result.status not in ("optimal", "feasible"):
        return result.status, None, None
    if result.x is None or "y" not in result.x:
        return result.status, None, None
    y_values = np.asarray(result.x["y"], dtype=np.float64).reshape(-1)
    x_objective = _posynomial_value(structure.objective, y_values)
    if not math.isfinite(x_objective):
        return result.status, None, None
    return result.status, x_objective, np.exp(y_values)


def _most_fractional_offset(
    x_flat: np.ndarray, integer_offsets: list[int], integrality_tol: float
) -> Optional[int]:
    """Return the discrete offset furthest from integral, or ``None`` if all integral."""
    worst_off: Optional[int] = None
    worst_frac = integrality_tol
    for off in integer_offsets:
        v = float(x_flat[off])
        frac = v - math.floor(v)
        dist = min(frac, 1.0 - frac)  # distance to the nearest integer
        if dist > worst_frac:
            worst_frac = dist
            worst_off = off
    return worst_off


def solve_gp_minlp(
    model: Model,
    *,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 100_000,
    integrality_tol: float = 1e-5,
    nlp_solver=None,
    ipopt_options=None,
) -> Optional[SolveResult]:
    """Solve a GP-structured MINLP by y-space node relaxations + integer B&B.

    Returns a :class:`SolveResult` in the original ``x``-space, or ``None`` if
    ``model`` is not a GP-structured MINLP (see :func:`classify_gp_minlp`).

    Method (issue #116). The continuous relaxation is a geometric program, so
    at each branch-and-bound node the box ``xᵢ ∈ [lᵢ, uᵢ]`` maps to
    ``yᵢ ∈ [log lᵢ, log uᵢ]`` and the node relaxation is solved as the *exact*
    convex log-space NLP — a rigorous bound (tighter than any ``x``-space
    relaxation of the non-convex posynomial problem, and a single convex solve).
    Branching is on the discrete variables in ``x``-space (floor/ceil of a
    fractional component); the recovered ``x = exp(y)`` maps incumbents back.

    Certification. Because every node bound is a rigorous convex-GP bound and
    the integer branching is exhaustive, a search that closes the tree returns a
    *certified* global optimum (``gap_certified=True``, ``gap == 0``). A run
    stopped by ``time_limit`` / ``max_nodes`` (or in which a node relaxation
    failed to converge) reports the best incumbent with the frontier's dual
    bound and ``gap_certified=False`` — never an over-claim.
    """
    info = classify_gp_minlp(model)
    if info is None:
        return None

    structure = info.structure
    minimize = structure.minimize
    integer_offsets = info.integer_offsets
    n = _total_scalars(model)
    x_lb0, x_ub0 = _x_bounds(model, n)

    # Work in an internal *minimisation* convention: internal = obj for a
    # minimise model, -obj for a maximise model. A GP maximise objective is a
    # single monomial, but the transform is generic. ``to_internal`` is its own
    # inverse (an involution), so it also maps an internal bound back to the
    # reported sense.
    def to_internal(v: float) -> float:
        return v if minimize else -v

    def fathom_slack() -> float:
        # A live/candidate node whose bound is within this of the incumbent
        # cannot improve it beyond the optimality tolerance, so it is fathomed.
        # Folds the relative ``gap_tolerance`` with a tiny absolute floor that
        # keeps the exact-solve (gap → 0) case pruning genuine ties only.
        return max(gap_tolerance * max(1.0, abs(best_internal)), _GP_MINLP_PRUNE_TOL)

    solve_kwargs: dict = {}
    if nlp_solver is not None:
        solve_kwargs["nlp_solver"] = nlp_solver
    if ipopt_options is not None:
        solve_kwargs["ipopt_options"] = ipopt_options

    start = time.time()
    deadline = start + time_limit if time_limit is not None else None

    best_internal = math.inf  # incumbent, internal (minimisation) sense
    best_obj: Optional[float] = None  # incumbent, original sense
    best_x_flat: Optional[np.ndarray] = None

    counter = itertools.count()
    # Heap entries: (priority, tiebreak, x_lb, x_ub). ``priority`` is a valid
    # internal lower bound for the node (its parent's solved bound; tightening
    # a box can only raise a convex minimum, so the parent bound bounds the
    # child too). ``-inf`` seeds the unbounded root.
    heap: list[tuple[float, int, np.ndarray, np.ndarray]] = [
        (-math.inf, next(counter), x_lb0, x_ub0)
    ]

    node_count = 0
    solve_failed = False
    # Smallest valid lower bound (internal sense) of any subtree we had to
    # abandon because its node relaxation did not converge. Such a subtree is no
    # longer represented on the frontier, so the global dual bound must fold in
    # its parent bound (``priority`` — a rigorous convex bound on that subtree)
    # or the incumbent would be mis-reported as a valid lower bound (unsound).
    abandoned_bound = math.inf
    limit_status: Optional[str] = None

    while heap:
        if node_count >= max_nodes:
            limit_status = "node_limit"
            break
        if deadline is not None and time.time() > deadline:
            limit_status = "time_limit"
            break

        priority, tiebreak, xl, xu = heapq.heappop(heap)
        # Best-first: ``priority`` is the smallest live lower bound, i.e. the
        # current global dual bound. Once the incumbent is within the optimality
        # tolerance of it, no live node can improve the incumbent enough to
        # matter → the whole frontier is fathomed. Push the node back first so
        # its bound still counts toward the reported global bound.
        if best_obj is not None and priority >= best_internal - fathom_slack():
            heapq.heappush(heap, (priority, tiebreak, xl, xu))
            break

        node_count += 1
        node_kwargs = dict(solve_kwargs)
        if deadline is not None:
            node_kwargs["time_limit"] = max(0.0, deadline - time.time())
        status, obj, x_flat = _solve_gp_node(structure, xl, xu, node_count, node_kwargs)

        if status == "infeasible":
            continue  # a rigorously infeasible node is pruned soundly
        if obj is None or x_flat is None:
            # A non-convergent relaxation yields no bound: we cannot certify
            # this subtree. Do not prune it as infeasible (unsound); record the
            # failure so the final result is reported uncertified, and keep the
            # parent bound (``priority``) as this abandoned subtree's dual bound
            # so the reported global bound stays a valid lower bound.
            solve_failed = True
            abandoned_bound = min(abandoned_bound, priority)
            continue

        internal = to_internal(obj)
        if best_obj is not None and internal >= best_internal - fathom_slack():
            continue  # node cannot improve the incumbent beyond tolerance

        frac_off = _most_fractional_offset(x_flat, integer_offsets, integrality_tol)
        if frac_off is None:
            # Integer-feasible relaxation optimum → a genuine incumbent; its
            # objective is exact (values are integral to tolerance).
            best_internal = internal
            best_obj = obj
            x_round = np.array(x_flat, dtype=np.float64)
            for off in integer_offsets:
                x_round[off] = float(np.round(x_round[off]))
            best_x_flat = x_round
            continue

        # Branch on the most-fractional discrete component. Each child must be
        # a *strict* tightening of the parent box — otherwise a value pinned
        # just outside the integer bound by solver tolerance could spawn a child
        # identical to its parent and loop forever. (``frac_off`` was already
        # confirmed non-integral to tolerance, so at least one child tightens.)
        v = float(x_flat[frac_off])
        floor_v = float(math.floor(v))
        ceil_v = float(math.ceil(v))
        children = []
        if floor_v < xu[frac_off]:  # down: xᵢ ≤ ⌊v⌋
            children.append((xl, _with(xu, frac_off, floor_v)))
        if ceil_v > xl[frac_off]:  # up: xᵢ ≥ ⌈v⌉
            children.append((_with(xl, frac_off, ceil_v), xu))
        for new_lb, new_ub in children:
            if new_lb[frac_off] > new_ub[frac_off] + 1e-12:
                continue  # empty child box
            heapq.heappush(heap, (internal, next(counter), new_lb, new_ub))

    # ── Assemble the result ────────────────────────────────────────────────
    # The search is *proven* (a certified verdict) when it ended by exhausting /
    # fathoming the frontier — no resource limit hit and no subtree abandoned to
    # a convergence failure. Then an incumbent is a global optimum (within the
    # optimality tolerance) and no-incumbent means rigorous infeasibility. This
    # rests on the inner convex solve reporting feasibility/optimality soundly —
    # the same trust :func:`solve_gp` places in the log-space convex solve.
    proven = limit_status is None and not solve_failed

    # Global dual bound (internal sense) = the smallest lower bound still
    # unresolved (open frontier nodes and any abandoned subtree), capped by the
    # incumbent (a dual bound can never exceed the incumbent for a minimisation).
    # With nothing unresolved and an incumbent, it floors at the incumbent (a
    # closed tree — zero gap).
    bound_candidates = [min((item[0] for item in heap), default=math.inf), abandoned_bound]
    if best_obj is not None:
        bound_candidates.append(best_internal)
    bound_internal = min(bound_candidates)

    objective: Optional[float] = best_obj
    bound: Optional[float] = to_internal(bound_internal) if math.isfinite(bound_internal) else None
    gap: Optional[float]
    if best_obj is not None:
        gap = abs(best_obj - bound) / max(1.0, abs(best_obj)) if bound is not None else None
        # "optimal" iff proven (gap already within tolerance by construction);
        # otherwise a best-so-far incumbent with a valid but uncertified gap.
        status = "optimal" if proven else "feasible"
    else:
        gap = None
        if proven:
            # Frontier exhausted with no feasible node → rigorously infeasible.
            status = "infeasible"
            bound = None
        else:
            # Stopped early with no incumbent: keep the frontier's valid dual
            # bound (informative) even though there is no gap to compute.
            status = (limit_status or "error") if solve_failed else (limit_status or "infeasible")

    certified = proven

    x_values = _x_flat_to_dict(model, best_x_flat) if best_x_flat is not None else None

    return SolveResult(
        status=status,
        objective=objective,
        bound=bound,
        gap=gap,
        x=x_values,
        node_count=node_count,
        wall_time=time.time() - start,
        convex_fast_path=False,
        gap_certified=certified,
        _model=model,
    )


def _with(arr: np.ndarray, offset: int, value: float) -> np.ndarray:
    """Return a copy of ``arr`` with ``arr[offset]`` replaced by ``value``."""
    out = np.array(arr, dtype=np.float64)
    out[offset] = value
    return out


__all__ = [
    "ConstraintLogCurvature",
    "GPConstraint",
    "GPMinlpStructure",
    "GPStructure",
    "GeometricProgram",
    "LogCurvature",
    "ModelLogCurvature",
    "ObjectiveLogCurvature",
    "as_geometric_program",
    "classify_gp",
    "classify_gp_minlp",
    "classify_log_curvature",
    "is_gp_minlp",
    "is_log_convex",
    "model_log_curvature",
    "solve_gp",
    "solve_gp_minlp",
]
