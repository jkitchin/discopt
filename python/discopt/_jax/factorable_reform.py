"""Factorable reformulation: rewrite nonlinear terms the relaxation pipeline
cannot handle natively into terms it CAN relax.

Two sound, value-preserving rewrites are applied (see issue #130):

1. **Sign-definite denominator clearing.** A constraint term ``N / D`` with a
   non-constant denominator ``D`` that is provably bounded away from zero over
   the variable box (``D > 0`` or ``D < 0`` by interval arithmetic) is cleared
   by multiplying the whole constraint through by ``D``.  The inequality sense
   is preserved when ``D > 0`` and flipped when ``D < 0``; equalities are
   unaffected.  This is exact: over the box ``D`` never vanishes, so the
   multiplied constraint has the identical solution set.

2. **Mixed repeated-factor product lifting.** A *pure polynomial* product such
   as ``x*x*y`` (= ``x**2 * y``) is not representable by the bilinear /
   trilinear / monomial relaxation pipeline (see
   ``term_classifier.classify_nonlinear_terms``: such terms fall into
   ``general_nl`` and are dropped from the MILP relaxation).  Each repeated
   power ``x**k`` (k >= 2) inside such a product is lifted to a fresh auxiliary
   variable ``w`` with the defining equality ``w == x**k`` (a monomial the
   pipeline *does* relax) and the product is rebuilt as ``w * y`` — a bilinear
   term the pipeline handles.  ``w == x**k`` reproduces the term value exactly,
   so the lifted model is equivalent to the original.

Both rewrites preserve the feasible set and the objective exactly; the only
effect on the *relaxation* is to expose a valid outer approximation where there
previously was none.  When neither rewrite applies the input model is returned
unchanged (zero overhead, zero behavioural change).  Anything the pass is not
certain it can rewrite soundly is left untouched, so it never regresses a model
that already solved.

**Convexity caveat.** Although value-preserving, both rewrites can turn a
*convex* model nonconvex — clearing ``x**2 / z`` (convex for ``z > 0``) yields
the bilinear ``x**2 - y*z``, and distributing a product breaks the structure
the convex fast path recognises.  The caller is therefore expected to gate this
pass to provably-nonconvex models (see ``has_factorable_work`` and the
convexity check in ``discopt.solver.solve_model``); the rewrite itself is
unconditional once invoked.
"""

from __future__ import annotations

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    UnaryOp,
    Variable,
    VarType,
)

from .gdp_reformulate import _bound_expression
from .term_classifier import _get_flat_index, distribute_products

# A denominator counts as sign-definite only when its interval is bounded away
# from zero by at least this margin — guards against a denominator that merely
# grazes zero (where the multiply-through would not be value-preserving).
_ZERO_MARGIN = 1e-9
# Reject an aux lift whose induced bound magnitude is effectively infinite: an
# unbounded monomial aux would make the relaxation/NLP unbounded.
_INF_THRESH = 1e15


def _leaf_index_and_exp(expr: Expression, model: Model):
    """If *expr* is a variable leaf or an integer power of one, return
    ``(leaf_expr, flat_index, exponent)``; otherwise ``None``."""
    if isinstance(expr, (Variable, IndexExpression)):
        idx = _get_flat_index(expr, model)
        return (expr, idx, 1) if idx is not None else None
    if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
        p = float(expr.right.value)
        if p == int(p) and int(p) >= 1:
            idx = _get_flat_index(expr.left, model)
            if idx is not None:
                return (expr.left, idx, int(p))
    return None


def _decompose_poly_product(expr: Expression, model: Model):
    """Walk a ``*``-tree and split it into ``(coeff, powers, extra)``.

    ``powers`` maps ``flat_index -> [leaf_expr, total_exponent]`` for variable
    factors; ``extra`` collects any non-polynomial factor (transcendental,
    division, ...).  Returns ``None`` if the node is not a multiplication tree
    (e.g. it contains a ``+``/``-``), so the caller recurses instead.
    """
    coeff = 1.0
    powers: dict[int, list] = {}
    extra: list[Expression] = []

    def visit(e: Expression) -> bool:
        nonlocal coeff
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, Constant) and e.value.ndim == 0:
            coeff *= float(e.value)
            return True
        leaf = _leaf_index_and_exp(e, model)
        if leaf is not None:
            leaf_expr, idx, exp = leaf
            if idx in powers:
                powers[idx][1] += exp
            else:
                powers[idx] = [leaf_expr, exp]
            return True
        # Any other factor (sqrt(...), exp(...), a/b, ...) is non-polynomial.
        extra.append(e)
        return True

    if not (isinstance(expr, BinaryOp) and expr.op == "*"):
        return None
    visit(expr)
    return coeff, powers, extra


def _needs_lift(powers: dict[int, list]) -> bool:
    """A pure polynomial product needs lifting iff it is a *mixed* product with
    a repeated factor — at least two distinct variables and some exponent >= 2
    (e.g. ``x*x*y``).  Single-variable monomials (``x**k``) and products of
    distinct variables (bilinear/trilinear/multilinear) are handled natively."""
    if len(powers) < 2:
        return False
    return any(exp >= 2 for _leaf, exp in powers.values())


class _Lifter:
    """Allocates monomial auxiliary variables ``w == leaf**k`` on *model*,
    deduplicating by (flat_index, exponent)."""

    def __init__(self, model: Model):
        self.model = model
        self._cache: dict[tuple[int, int], Variable] = {}
        self._expr_cache: dict[int, Variable] = {}
        self.aux_constraints: list[Constraint] = []
        self._counter = 0

    def monomial(self, leaf: Expression, flat_index: int, exp: int):
        """Return an aux variable equal to ``leaf**exp`` (creating it on first
        use), or ``None`` if a finite bound for it cannot be established."""
        key = (flat_index, exp)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        pow_expr = BinaryOp("**", leaf, Constant(float(exp)))
        lo, hi = _bound_expression(pow_expr, self.model)
        if not (np.isfinite(lo) and np.isfinite(hi)) or max(abs(lo), abs(hi)) >= _INF_THRESH:
            return None
        name = f"_fr_aux_{self._counter}"
        self._counter += 1
        w = Variable(name, VarType.CONTINUOUS, (), lo, hi, self.model)
        self.model._variables.append(w)
        self._cache[key] = w
        # Normalised form ``body == 0`` with body = w - leaf**exp.
        self.aux_constraints.append(Constraint(BinaryOp("-", w, pow_expr), "==", 0.0))
        return w

    def expression(self, expr: Expression):
        """Return an aux variable equal to *expr* (creating it on first use), or
        ``None`` if a finite bound for it cannot be established.

        Used to expose a fractional power ``base ** p`` of a composite *base* to
        the relaxation: the relaxation can bound ``t ** p`` (a fractional power of
        a single variable) but not ``base ** p`` (a fractional power of a
        polynomial). The defining equality ``t == base`` is itself run through the
        monomial lift so any mixed products inside *base* become bilinear aux too.
        Deduplicated by structural identity of the (already-distributed) node.
        """
        key = id(expr)
        cached = self._expr_cache.get(key)
        if cached is not None:
            return cached
        lo, hi = _bound_expression(expr, self.model)
        if not (np.isfinite(lo) and np.isfinite(hi)) or max(abs(lo), abs(hi)) >= _INF_THRESH:
            return None
        name = f"_fr_aux_{self._counter}"
        self._counter += 1
        w = Variable(name, VarType.CONTINUOUS, (), lo, hi, self.model)
        self.model._variables.append(w)
        self._expr_cache[key] = w
        # Clear any sign-definite division in the defining equality ``w == expr``
        # (so a ratio ``w == N / D`` becomes the bilinear ``w*D == N`` the
        # relaxation can McCormick), then lift its mixed products.
        body, _sense = _clear_divisions(BinaryOp("-", w, expr), "==", self.model)
        body = _lift_expr(distribute_products(body), self.model, self)
        self.aux_constraints.append(Constraint(body, "==", 0.0))
        return w

    def fractional_power(self, base_var: Variable, p: float):
        """Return an aux variable equal to ``base_var ** p`` for a fractional *p*,
        or ``None`` if it cannot be soundly bounded.

        ``base_var`` is a single variable (typically an aux from :meth:`expression`
        holding a polynomial), so ``base_var ** p`` is a fractional power of a
        *variable* — which the relaxation can bound. Lifting the *value* of the
        power (not just its base) turns e.g. ``N / g**(1/3)`` into ``N / d`` with
        ``d`` a plain variable, the ratio form the objective linearizer accepts.
        Requires ``base_var >= 0`` so the power is real and monotone increasing.
        """
        lo = float(np.min(base_var.lb))
        hi = float(np.max(base_var.ub))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0.0:
            return None
        d_lo, d_hi = lo**p, hi**p
        if not (np.isfinite(d_lo) and np.isfinite(d_hi)) or max(d_lo, d_hi) >= _INF_THRESH:
            return None
        name = f"_fr_aux_{self._counter}"
        self._counter += 1
        d = Variable(name, VarType.CONTINUOUS, (), d_lo, d_hi, self.model)
        self.model._variables.append(d)
        pow_expr = BinaryOp("**", base_var, Constant(float(p)))
        self.aux_constraints.append(Constraint(BinaryOp("-", d, pow_expr), "==", 0.0))
        return d


def _rebuild_product(coeff: float, atoms: list[Expression]) -> Expression:
    """Reconstruct ``coeff * atoms[0] * atoms[1] * ...`` as a left-folded tree."""
    expr: Expression | None = None
    if coeff != 1.0 or not atoms:
        expr = Constant(coeff)
    for a in atoms:
        expr = a if expr is None else BinaryOp("*", expr, a)
    return expr if expr is not None else Constant(coeff)


def _lift_expr(expr: Expression, model: Model, lifter: _Lifter) -> Expression:
    """Return *expr* with every mixed repeated-factor polynomial product lifted
    to bilinear form via monomial aux variables.  Identity-preserving: returns
    the same object when nothing changed, so untouched subtrees are unaffected.
    """
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            decomp = _decompose_poly_product(expr, model)
            if decomp is not None:
                coeff, powers, extra = decomp
                # Only a pure polynomial product (no transcendental/division
                # factor) is liftable into supported bilinear terms.
                if not extra and _needs_lift(powers):
                    atoms: list[Expression] = []
                    ok = True
                    for _idx, (leaf, exp) in powers.items():
                        if exp >= 2:
                            w = lifter.monomial(leaf, _idx, exp)
                            if w is None:
                                ok = False
                                break
                            atoms.append(w)
                        else:
                            atoms.append(leaf)
                    if ok:
                        return _rebuild_product(coeff, atoms)
            # Not a liftable product — recurse into factors.
        left = _lift_expr(expr.left, model, lifter)
        right = _lift_expr(expr.right, model, lifter)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _lift_expr(expr.operand, model, lifter)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    # Do not descend into FunctionCall args: lifting inside a transcendental
    # does not help the relaxation (the call is general_nl regardless) and must
    # not disturb composite/univariate handling of e.g. sqrt(x**2 + c).
    return expr


def _is_simple_power_base(expr: Expression, model: Model) -> bool:
    """A power base the relaxation already handles directly: a variable leaf or
    an integer power of one (the monomial path), so it must not be lifted."""
    return _leaf_index_and_exp(expr, model) is not None


def _lift_objective_atoms(expr: Expression, model: Model, lifter: "_Lifter") -> Expression:
    """Decompose composite fractional-power and variable/variable-ratio atoms into
    elementary auxiliary variables, so an objective the relaxation would otherwise
    drop becomes linear in supported terms.

    The relaxation can bound a fractional power of a *single variable*
    (``fractional_power_var_map``) and a sign-definite ratio (via clearing), but
    not a fractional power of a polynomial or a ratio whose value sits raw in the
    objective (the objective linearizer rejects a non-constant division). So,
    bottom-up:

    * ``base ** p`` (non-integer *p*) -> a plain aux ``d == t ** p`` where ``t``
      is *base* lifted to a variable. Exposes ``N / d`` instead of ``N / base**p``.
    * ``N / D`` (non-constant *D*) -> a plain aux ``r == N / D`` (whose defining
      equality is cleared to the bilinear ``r*D == N``).

    Recursion composes these: ``(N / g**(1/3))**0.83`` (st_e35) becomes
    ``g->t, t**(1/3)->d, N/d->r, r**0.83->s`` and ``N / g**(1/3)`` (ex1233) becomes
    ``g->t, t**(1/3)->d, N/d->r``, leaving the objective linear in the aux.
    Identity-preserving; leaves a node untouched when an operand has no finite
    interval (e.g. an unbounded variable), so the rewrite is never unsound — at
    worst the term stays dropped, exactly as before.
    """
    if isinstance(expr, BinaryOp):
        if (
            expr.op == "**"
            and isinstance(expr.right, Constant)
            and float(expr.right.value) != int(float(expr.right.value))
        ):
            p = float(expr.right.value)
            base = _lift_objective_atoms(expr.left, model, lifter)
            # The fractional power needs a single-variable argument ``t``.
            if isinstance(base, (Variable, IndexExpression)):
                t = base
            else:
                t = lifter.expression(distribute_products(base))
            if isinstance(t, Variable):
                d = lifter.fractional_power(t, p)
                if d is not None:
                    return d
            if base is not expr.left:
                return BinaryOp("**", base, expr.right)
            return expr
        if expr.op == "/" and not isinstance(expr.right, Constant):
            num = _lift_objective_atoms(expr.left, model, lifter)
            den = _lift_objective_atoms(expr.right, model, lifter)
            ratio = expr if (num is expr.left and den is expr.right) else BinaryOp("/", num, den)
            r = lifter.expression(distribute_products(ratio))
            if r is not None:
                return r
            return ratio
        left = _lift_objective_atoms(expr.left, model, lifter)
        right = _lift_objective_atoms(expr.right, model, lifter)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _lift_objective_atoms(expr.operand, model, lifter)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    return expr


def _find_clearable_denominator(expr: Expression, model: Model):
    """Return the denominator ``D`` of the first division term ``N/D`` in
    *expr*'s additive structure whose ``D`` is non-constant and sign-definite
    over the variable box, as ``(D, sign, dmin)`` where ``sign`` is +1/-1 and
    ``dmin = min |D|`` over the box.  ``None`` if no such division exists."""
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            found = _find_clearable_denominator(expr.left, model)
            if found is not None:
                return found
            return _find_clearable_denominator(expr.right, model)
        if expr.op == "/":
            d = expr.right
            if not isinstance(d, Constant):
                lo, hi = _bound_expression(d, model)
                if lo > _ZERO_MARGIN:
                    return d, 1, lo
                if hi < -_ZERO_MARGIN:
                    return d, -1, -hi
            # Search the numerator for a nested division.
            return _find_clearable_denominator(expr.left, model)
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _find_clearable_denominator(expr.operand, model)
    return None


def _multiply_through(expr: Expression, denom: Expression) -> Expression:
    """Return ``expr * denom`` with any division by *denom* (same object)
    cancelled, distributing the multiply over the additive structure."""
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            return BinaryOp(
                expr.op,
                _multiply_through(expr.left, denom),
                _multiply_through(expr.right, denom),
            )
        if expr.op == "/" and expr.right is denom:
            return expr.left  # (N / D) * D -> N
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return UnaryOp("neg", _multiply_through(expr.operand, denom))
    # ``0 * D`` is just 0 — keep the rewritten body free of dead zero terms.
    if isinstance(expr, Constant) and expr.value.ndim == 0 and float(expr.value) == 0.0:
        return expr
    return BinaryOp("*", expr, denom)


_FLIP = {"<=": ">=", ">=": "<=", "==": "=="}


def _clear_divisions(body: Expression, sense: str, model: Model):
    """Clear every sign-definite denominator from a constraint ``body sense 0``.
    Returns ``(new_body, new_sense)``.

    Multiplying a constraint through by a denominator ``D`` rescales it by
    ``D(x)``.  When ``|D|`` can be < 1 over the box, a *gross* violation of the
    original constraint shrinks proportionally in the cleared form — e.g.
    clearing ``6 - x0 + 0.2458 x0**2/x1 <= 0`` by ``x1 in [1e-5, 30]`` turns a
    violation of 6.0 into ``6.0 * x1 ~ 6e-5``, which then slips *under* the
    absolute incumbent-feasibility tolerance (1e-4).  The spatial-B&B would then
    accept an infeasible point as a feasible incumbent and certify it — a
    false-optimal.  To keep the fixed absolute tolerance sound, divide the
    cleared body by ``dmin = min |D|`` so the scaled magnitude is never *smaller*
    than the original (``|D(x)| / dmin >= 1`` everywhere in the box).  This is
    exact (division by a positive constant preserves the feasible set) and only
    ever makes the feasibility test stricter, never looser.
    """
    scale = 1.0
    for _ in range(8):  # bounded: each pass clears one denominator family
        found = _find_clearable_denominator(body, model)
        if found is None:
            break
        denom, sign, dmin = found
        body = _multiply_through(body, denom)
        if sign < 0:
            sense = _FLIP[sense]
        if dmin < 1.0:
            scale /= dmin
    if scale != 1.0:
        body = BinaryOp("*", Constant(scale), body)
    return body, sense


def has_factorable_work(model: Model) -> bool:
    """True if any constraint/objective has a clearable division or a mixed
    repeated-factor product — i.e. the pass would change the model.

    Exposed so the solver can run this cheap structural scan *before* the more
    expensive convexity classification used to gate the (convexity-destroying)
    rewrite: only nonconvex models that actually have liftable terms pay for
    convexity detection.
    """

    def scan(expr: Expression) -> bool:
        if _find_clearable_denominator(expr, model) is not None:
            return True
        dist = distribute_products(expr)
        return _scan_for_mixed_product(dist, model)

    if model._objective is not None and scan(model._objective.expression):
        return True
    return any(isinstance(c, Constraint) and scan(c.body) for c in model._constraints)


def has_clearable_denominator(model: Model) -> bool:
    """True if any constraint has a sign-definite, non-constant denominator that
    denominator clearing would rewrite.

    Distinct from :func:`has_factorable_work`, which also fires on mixed
    repeated-factor products.  The solver uses this to decide whether a
    *convex* model is worth clearing: a non-constant division drops to
    ``general_nl`` and so cannot be bounded by the relaxation (no dual bound, no
    certification), and clearing a sign-definite denominator is exact — so it is
    a strict improvement for such models, unlike the mixed-product lift which can
    only destroy convexity.  Objective ratios are excluded: clearing multiplies a
    *constraint* through by its denominator and has no analogue for an objective.
    """
    return any(
        isinstance(c, Constraint) and _find_clearable_denominator(c.body, model) is not None
        for c in model._constraints
    )


def _scan_for_mixed_product(expr: Expression, model: Model) -> bool:
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            decomp = _decompose_poly_product(expr, model)
            if decomp is not None:
                _coeff, powers, extra = decomp
                if not extra and _needs_lift(powers):
                    return True
        return _scan_for_mixed_product(expr.left, model) or _scan_for_mixed_product(
            expr.right, model
        )
    if isinstance(expr, UnaryOp):
        return _scan_for_mixed_product(expr.operand, model)
    return False


def factorable_reformulate(model: Model, *, clear_only: bool = False) -> Model:
    """Return a model equivalent to *model* with sign-definite denominators
    cleared and mixed repeated-factor products lifted to bilinear form.

    If neither rewrite applies, *model* is returned unchanged.  On any
    unexpected error the original model is returned, so the pass can never make
    a previously-solvable model unsolvable.

    ``clear_only`` restricts the pass to denominator clearing and skips the
    mixed repeated-factor product lift entirely.  The lift distributes products
    and introduces ``w == x**k`` aux variables, which destroys convex structure
    even where it was unnecessary; clearing alone is the right rewrite for a
    *convex* model that merely needs its non-constant division exposed to the
    relaxation (see ``has_clearable_denominator``).
    """
    try:
        if not has_factorable_work(model):
            return model

        new_model = Model(model.name)
        new_model._variables = list(model._variables)
        new_model._parameters = list(model._parameters)
        new_model._objective = model._objective

        lifter = _Lifter(new_model)

        rebuilt: list[Constraint] = []
        for c in model._constraints:
            if not isinstance(c, Constraint):
                rebuilt.append(c)  # pass through anything exotic untouched
                continue
            body, sense = _clear_divisions(c.body, c.sense, new_model)
            if clear_only:
                # Only touch constraints the clearing actually rewrote; leave
                # everything else byte-for-byte identical so convex structure
                # elsewhere is preserved.
                if body is c.body and sense == c.sense:
                    rebuilt.append(c)
                else:
                    rebuilt.append(Constraint(distribute_products(body), sense, c.rhs, c.name))
                continue
            body = distribute_products(body)
            body = _lift_objective_atoms(body, new_model, lifter)
            body = _lift_expr(body, new_model, lifter)
            if body is c.body and sense == c.sense:
                rebuilt.append(c)
            else:
                rebuilt.append(Constraint(body, sense, c.rhs, c.name))

        # Lift the objective too (it may contain a mixed product or a fractional
        # power of a polynomial base); division clearing is meaningless for an
        # objective so only the lifts apply.
        if not clear_only and new_model._objective is not None:
            obj_expr = distribute_products(new_model._objective.expression)
            obj_expr = _lift_objective_atoms(obj_expr, new_model, lifter)
            lifted_obj = _lift_expr(obj_expr, new_model, lifter)
            if lifted_obj is not new_model._objective.expression:
                from discopt.modeling.core import Objective

                new_model._objective = Objective(lifted_obj, new_model._objective.sense)

        # Defining equalities for the aux variables come first so downstream
        # bound propagation sees them early.
        new_model._constraints = lifter.aux_constraints + rebuilt
        return new_model
    except Exception:  # pragma: no cover - defensive: never break a solve
        return model
