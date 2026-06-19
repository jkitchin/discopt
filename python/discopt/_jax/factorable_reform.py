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
    FunctionCall,
    IndexExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

from .gdp_reformulate import _bound_expression, _collect_variables, _is_linear
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
        # Keyed by a STRUCTURAL representation of the expression, never ``id()``:
        # CPython recycles the ``id()`` of a garbage-collected object, so a later,
        # structurally *different* expression can reuse a freed address and score a
        # false cache hit — returning a stale aux for the wrong sub-expression.
        # In ex7_2_3 that dropped a ``/x8`` denominator from a lifted ratio,
        # producing a non-feasibility-preserving reformulation that certified an
        # infeasible box corner as the global optimum (a false "optimal"). A
        # structural key can only ever *fail* to dedup (harmless — an extra aux),
        # never falsely merge two distinct expressions.
        self._expr_cache: dict[tuple[str, float | None], Variable] = {}
        self.aux_constraints: list[Constraint] = []
        self._counter = 0

    def monomial(self, leaf: Expression, flat_index: int, exp: int) -> Variable | None:
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

    def expression(self, expr: Expression, *, lb_floor: float | None = None) -> Variable | None:
        """Return an aux variable equal to *expr* (creating it on first use), or
        ``None`` if a finite bound for it cannot be established.

        Used to expose a fractional power ``base ** p`` of a composite *base* to
        the relaxation: the relaxation can bound ``t ** p`` (a fractional power of
        a single variable) but not ``base ** p`` (a fractional power of a
        polynomial). The defining equality ``t == base`` is itself run through the
        monomial lift so any mixed products inside *base* become bilinear aux too.
        Deduplicated by structural identity of the (already-distributed) node.

        *lb_floor*, when given, raises the aux lower bound to ``max(lo, lb_floor)``.
        Used by the transcendental-argument lift: ``sqrt(g)`` is real only where
        ``g >= 0``, so flooring the aux for ``t == g`` at 0 is feasibility-
        preserving and is required for the univariate sqrt envelope to apply (it
        abstains on an argument whose lower bound is negative). The floor is part
        of the cache key so a floored aux is never reused where an unfloored one
        is expected (which would unsoundly tighten the other use site).
        """
        key = (repr(expr), lb_floor)
        cached = self._expr_cache.get(key)
        if cached is not None:
            return cached
        lo, hi = _bound_expression(expr, self.model)
        if lb_floor is not None:
            lo = max(lo, lb_floor)
        if not (np.isfinite(lo) and np.isfinite(hi)) or max(abs(lo), abs(hi)) >= _INF_THRESH:
            return None
        if lo > hi:
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

    def fractional_power(self, base_var: Variable, p: float) -> Variable | None:
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


# Distributing a product whose factors are multi-term sums multiplies their term
# counts.  A product of several sum-of-squares (st_e36: 5 factors → a 12.6 MB,
# degree-~17 polynomial when distributed) both explodes memory/time and forces
# high-degree univariate monomial lifts (``x1**8`` over [15,25] is 1.5e11, past
# the relaxation's 1e10 aux-bound limit) whose columns are then dropped — gutting
# the relaxation and leaving the constraint effectively unrelaxed.  The standard
# factorable cure (BARON, Couenne) is to NOT distribute: lift each multi-term
# factor to a bounded aux ``w_i == f_i`` and relax the resulting bilinear /
# multilinear product of variables directly.
_DISTRIBUTE_TERM_LIMIT = 1024  # est. distributed-term count above which a product is lifted
_TERM_CAP = 1 << 40  # saturate the estimate so deep blowups can't overflow


def _estimate_distributed_terms(expr: Expression) -> int:
    """Estimate how many additive terms ``distribute_products(expr)`` would yield.

    A product multiplies its operands' term counts, an integer power raises the
    base's, a sum adds them; every opaque leaf (variable, call, division,
    constant) counts as one.  Saturated at ``_TERM_CAP`` so a deeply nested
    blowup cannot overflow before it trips the limit.
    """
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            return min(
                _estimate_distributed_terms(expr.left)
                + _estimate_distributed_terms(expr.right),
                _TERM_CAP,
            )
        if expr.op == "*":
            return min(
                _estimate_distributed_terms(expr.left)
                * _estimate_distributed_terms(expr.right),
                _TERM_CAP,
            )
        if expr.op == "**" and isinstance(expr.right, Constant):
            n = float(expr.right.value)
            n_int = int(n)
            if n == n_int and n_int >= 1:
                return min(_estimate_distributed_terms(expr.left) ** n_int, _TERM_CAP)
        return 1
    if isinstance(expr, UnaryOp):
        return _estimate_distributed_terms(expr.operand)
    return 1


def _collect_mul_factors(expr: Expression) -> list[Expression]:
    """Flatten a left/right-nested ``*`` chain into its factor list."""
    if isinstance(expr, BinaryOp) and expr.op == "*":
        return _collect_mul_factors(expr.left) + _collect_mul_factors(expr.right)
    return [expr]


def _prelift_blowup_products(expr: Expression, model: Model, lifter: "_Lifter") -> Expression:
    """Lift the factors of any product whose naive distribution would explode.

    Walks *expr*; at each ``*``-rooted product whose estimated distributed term
    count exceeds :data:`_DISTRIBUTE_TERM_LIMIT`, lifts every multi-term-sum
    factor ``f`` to an aux ``w == f`` (an exact equality) and rebuilds the
    product over the auxes, so the relaxation sees a bilinear/multilinear product
    of bounded variables instead of a high-degree expanded polynomial.  Sound:
    each ``w == f`` is exact and the lifted product is McCormick-relaxable.
    Identity-preserving for every node under the limit, so non-blowup
    constraints/objectives are byte-for-byte unchanged.
    """
    if isinstance(expr, BinaryOp):
        if expr.op == "*" and _estimate_distributed_terms(expr) > _DISTRIBUTE_TERM_LIMIT:
            new_factors: list[Expression] = []
            changed = False
            for f in _collect_mul_factors(expr):
                # Only a genuine multi-term sum drives the blowup; a constant,
                # variable, or monomial power distributes to one term and is left
                # for the normal monomial/bilinear path.
                if (
                    isinstance(f, BinaryOp)
                    and f.op in ("+", "-")
                    and _estimate_distributed_terms(f) >= 2
                ):
                    # Recurse first so a factor that is *itself* a blowup product
                    # has its inner factors lifted before this one is bounded.
                    f_lifted = _prelift_blowup_products(f, model, lifter)
                    w = lifter.expression(f_lifted)
                    if w is not None:
                        new_factors.append(w)
                        changed = True
                        continue
                    new_factors.append(f_lifted)
                else:
                    new_factors.append(_prelift_blowup_products(f, model, lifter))
            if changed:
                return _rebuild_product(1.0, new_factors)
            # Couldn't bound any factor (unbounded box): leave the product to the
            # existing distribute/monomial path rather than alter it.
            return expr
        left = _prelift_blowup_products(expr.left, model, lifter)
        right = _prelift_blowup_products(expr.right, model, lifter)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _prelift_blowup_products(expr.operand, model, lifter)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    return expr


# Univariate transcendentals whose argument may be lifted into an aux variable.
# ``sqrt``/``exp`` are domain-safe to relax over a finite box once their argument
# is a single variable (sqrt needs ``arg >= 0``, supplied by the lb floor below;
# exp is defined on all reals).  ``log``-family is excluded: it needs a strictly
# positive argument lower bound that interval arithmetic does not always certify.
_LIFTABLE_CALL_OUTER = {"sqrt", "exp"}


def _should_lift_call_arg(call: Expression, model: Model) -> Expression | None:
    """If *call* is a univariate transcendental over a non-affine, multivariate
    argument whose whole-node curvature is UNKNOWN, return its argument (the
    sub-expression to lift); otherwise ``None``.

    Lifting ``outer(g(x))`` -> ``outer(t)`` with ``t == g(x)`` decomposes a node
    the relaxation drops (a transcendental over a factorable polynomial with
    cross terms — e.g. ``sqrt(x4**2 + 2*x4*x5*x7 + x5**2)``) into a polynomial
    equality (relaxed by the existing McCormick/RLT machinery) plus a univariate
    envelope (the existing secant/tangent relaxation of ``outer``).

    Gated so it never *downgrades* a node the pipeline already relaxes tightly:

    * Affine argument -> skip: ``outer`` of an affine expression is already
      handled directly by the univariate envelope; an aux buys nothing.
    * Single-variable argument -> skip: ``sqrt(x**2 + c)`` is reached by the
      composite-univariate path; the lift targets genuinely multivariate args.
    * Proven CONVEX/CONCAVE node -> skip: e.g. an affine 2-norm ``sqrt(xᵀQx)``
      recognised by the convexity detector is relaxed exactly by its own tight
      envelope; lifting would replace that with the looser concave-sqrt secant.
    """
    if not isinstance(call, FunctionCall):
        return None
    if call.func_name not in _LIFTABLE_CALL_OUTER or len(call.args) != 1:
        return None
    arg = call.args[0]
    if _is_linear(arg):
        return None
    if len(_collect_variables(arg)) < 2:
        return None
    from .convexity.lattice import Curvature
    from .convexity.rules import classify_expr

    if classify_expr(call, model) != Curvature.UNKNOWN:
        return None
    return arg


def _simplify_sqrt_monomial(call: Expression, model: Model) -> Expression | None:
    """Rewrite ``sqrt(c * prod x_i**e_i)`` via the root-of-perfect-power identity.

    On a non-negative domain ``sqrt(x**(2k)) == x**k`` *exactly*, so a ``sqrt``
    whose argument is a monomial with all-even exponents and a non-negative
    constant factor collapses to a plain monomial — no transcendental, no
    envelope at all. Being an algebraic identity (not a relaxation) it is sound
    for any curvature, and it sidesteps the extreme-magnitude abstention that
    otherwise *drops the whole constraint*: nvs05's omitted
    ``sqrt(1e15*x2**2*x3**6)`` becomes ``3.16e7*x2*x3**3`` — the 2.5e33-wide
    argument turns into a 3.16e7 coefficient on a benign monomial the existing
    monomial/bilinear machinery relaxes tightly.

    The all-even / non-negative-base requirement is necessary, not conservative:
    ``sqrt(x**(2k)) == |x|**k``, which equals ``x**k`` only when ``x >= 0`` (or
    ``k`` even). Returns the rewritten monomial, or ``None`` when the argument is
    not a non-negative all-even-exponent monomial (the caller then tries the
    general aux-variable lift).
    """
    if not isinstance(call, FunctionCall):
        return None
    if call.func_name != "sqrt" or len(call.args) != 1:
        return None
    decomp = _decompose_poly_product(distribute_products(call.args[0]), model)
    if decomp is None:
        return None
    coeff, powers, extra = decomp
    if extra or not powers or coeff < 0.0:
        return None
    atoms: list[Expression] = []
    for _idx, (leaf, exp) in powers.items():
        if exp % 2 != 0:
            return None  # residual odd power -> argument is not a perfect square
        lo, _hi = _bound_expression(leaf, model)
        if not np.isfinite(lo) or lo < 0.0:
            return None  # sqrt(leaf**exp) == |leaf|**(exp/2) != leaf**(exp/2) for lo<0
        half = exp // 2
        atoms.append(leaf if half == 1 else BinaryOp("**", leaf, Constant(float(half))))
    return _rebuild_product(float(np.sqrt(coeff)), atoms)


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
    if isinstance(expr, FunctionCall):
        # Root-of-perfect-power simplification first: ``sqrt(c * perfect-square
        # monomial)`` is an exact monomial, strictly better than any envelope and
        # immune to the extreme-magnitude abstention. Recurse so the resulting
        # monomial gets its own mixed-product lift (e.g. ``x2*x3**3`` -> bilinear
        # ``x2 * aux(x3**3)``).
        simplified = _simplify_sqrt_monomial(expr, model)
        if simplified is not None:
            return _lift_expr(simplified, model, lifter)
        # Auxiliary-variable factorization of ``outer(g(x))`` (issue #130 follow-up):
        # when ``g`` is a multivariate non-affine argument the univariate envelope
        # cannot reach (cross terms), lift ``t == g(x)`` and rewrite the node as
        # ``outer(t)``.  The defining equality is itself run through the lift inside
        # ``_Lifter.expression`` so any mixed products in ``g`` become bilinear aux.
        # For sqrt the aux is floored at 0 (``sqrt`` is real only where ``g >= 0``),
        # which is feasibility-preserving and lets the univariate sqrt envelope
        # apply.  Gated by ``_should_lift_call_arg`` so a node already proven convex
        # (an affine 2-norm) keeps its tight envelope rather than being downgraded.
        arg = _should_lift_call_arg(expr, model)
        if arg is not None:
            lb_floor = 0.0 if expr.func_name == "sqrt" else None
            t = lifter.expression(distribute_products(arg), lb_floor=lb_floor)
            if t is not None:
                return FunctionCall(expr.func_name, t)
        # Otherwise do not descend into FunctionCall args: lifting inside a
        # transcendental does not help the relaxation (the call is general_nl
        # regardless) and must not disturb composite/univariate handling of e.g.
        # sqrt(x**2 + c).
        return expr
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
            t: Variable | IndexExpression | None
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


def _has_unbounded_nonlinear_term(body: Expression, model: Model) -> bool:
    """True if the distributed *body* contains a nonlinear product term (total
    variable degree >= 2) whose interval bound is non-finite over the model box.

    Denominator clearing multiplies the *whole* constraint through by ``D``, so a
    benign linear term in an unbounded variable (e.g. a continuous slack ``x4``
    with ``ub = +inf``) becomes a nonlinear product ``x4 * D``.  A product with a
    non-finitely-bounded factor has no valid finite McCormick/bilinear envelope:
    the relaxation built on it can *exclude* feasible points and report a false
    infeasibility (gear4, a feasible MINLPLib instance whose linear slacks are
    unbounded above).  When clearing introduces such a term the rewrite must be
    rejected and the original quotient kept — the McCormick-``lp`` path bounds the
    division soundly.
    """
    dist = distribute_products(body)

    def walk(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op in ("+", "-"):
            return walk(e.left) or walk(e.right)
        if isinstance(e, UnaryOp) and e.op == "neg":
            return walk(e.operand)
        decomp = _decompose_poly_product(e, model)
        if decomp is not None:
            _coeff, powers, extra = decomp
            total_degree = sum(exp for _leaf, exp in powers.values())
            if not extra and total_degree >= 2:
                lo, hi = _bound_expression(e, model)
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    return True
        return False

    return walk(dist)


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
        if _scan_for_mixed_product(dist, model):
            return True
        return _scan_for_liftable_call(dist, model)

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


def _scan_for_liftable_call(expr: Expression, model: Model) -> bool:
    """True if *expr* contains a transcendental node whose argument the
    auxiliary-variable factorization would lift (see ``_should_lift_call_arg``)."""
    if isinstance(expr, FunctionCall):
        if _should_lift_call_arg(expr, model) is not None:
            return True
        return any(_scan_for_liftable_call(a, model) for a in expr.args)
    if isinstance(expr, BinaryOp):
        return _scan_for_liftable_call(expr.left, model) or _scan_for_liftable_call(
            expr.right, model
        )
    if isinstance(expr, UnaryOp):
        return _scan_for_liftable_call(expr.operand, model)
    return False


# ---------------------------------------------------------------------------
# Entropy canonicalization: x*log(x) -> entropy(x)
# ---------------------------------------------------------------------------
#
# AMPL/GAMS lower the ``entropy`` intrinsic into a raw ``x*log(x)`` product
# when they emit a ``.nl`` file, so a model whose objective is ``Σ xᵢ·log(xᵢ)``
# (chemical-equilibrium / Gibbs free energy, e.g. globallib ``ex6_1_4``) reaches
# the relaxer as an undecomposable product. The MILP/McCormick-LP relaxer cannot
# decompose ``x*log(x)`` and falls back to a *constant* separable objective floor
# that never tightens under branching — discopt finds the global optimum but
# cannot certify it (issue #207).
#
# discopt already carries a dedicated convex underestimator for the ``entropy``
# intrinsic (``entropy(x) = x*log(x)``, convex on ``x ≥ 0``): ``relax_entropy``
# (mccormick.py), the curvature lattice (convexity), and interval arithmetic all
# recognise it. Recovering the intrinsic from the lowered product — a pure DAG
# canonicalization — is therefore enough to feed the existing relaxation, so the
# objective bound tightens under branching (and a separable-entropy objective is
# detected as convex, unlocking the convex fast path). The rewrite is exact:
# ``entropy(x)`` and ``x*log(x)`` are the same function, so it is sound for any
# model, convex or not, and so runs unconditionally.


def _strip_neg(expr: Expression) -> tuple[float, Expression]:
    """Peel leading ``neg(...)`` wrappers, returning ``(sign, inner)``."""
    sign = 1.0
    while isinstance(expr, UnaryOp) and expr.op == "neg":
        sign = -sign
        expr = expr.operand
    return sign, expr


def _flatten_product(expr: Expression) -> list[Expression]:
    """Flatten a left/right-nested ``*``-tree into a flat list of factors."""
    if isinstance(expr, BinaryOp) and expr.op == "*":
        return _flatten_product(expr.left) + _flatten_product(expr.right)
    return [expr]


def _match_entropy_product(expr: Expression, model: Model) -> Expression | None:
    """If *expr* is a product equal to ``c · x · log(x)`` for a single variable
    ``x`` with a nonnegative, finite box, return ``c · entropy(x)``; else ``None``.

    The match is deliberately strict: it fires only when, after folding constant
    factors, the product's variable content is *exactly* one bare occurrence of a
    variable ``x`` and one ``log(x)`` of the same variable (so ``x²·log(x)``,
    ``x·log(a·x)``, ``x·log(y)`` etc. are left untouched). The domain guard
    (``lb ≥ 0``, finite box) keeps the substitution consistent with
    ``relax_entropy``'s requirement and never introduces ``entropy`` where the
    original product was outside the entropy domain.
    """
    coeff = 1.0
    log_idx: int | None = None
    log_arg: Expression | None = None
    bare_idx: int | None = None
    bare_count = 0
    log_count = 0

    for factor in _flatten_product(expr):
        sign, factor = _strip_neg(factor)
        coeff *= sign
        # Fold scalar-constant factors into the coefficient.
        if isinstance(factor, Constant) and factor.value.ndim == 0:
            coeff *= float(factor.value)
            continue
        # log(var)?
        if isinstance(factor, FunctionCall) and factor.func_name == "log" and len(factor.args) == 1:
            idx = _get_flat_index(factor.args[0], model)
            if idx is not None:
                log_count += 1
                log_idx = idx
                log_arg = factor.args[0]
                continue
            return None
        # bare variable leaf?
        idx = _get_flat_index(factor, model)
        if idx is not None:
            bare_count += 1
            bare_idx = idx
            continue
        # Any other factor (transcendental, product of vars, ...) disqualifies.
        return None

    if log_count != 1 or bare_count != 1 or log_arg is None or log_idx != bare_idx:
        return None

    # Domain guard: entropy's underestimator requires a nonnegative, finite box.
    lo, hi = _bound_expression(log_arg, model)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0.0:
        return None

    entropy = FunctionCall("entropy", log_arg)
    if coeff == 1.0:
        return entropy
    return BinaryOp("*", Constant(coeff), entropy)


def _match_centropy_product(expr: Expression, model: Model) -> Expression | None:
    """If *expr* is a product equal to ``c · x · log(x/y)`` for a single variable
    ``x`` (nonnegative, finite box) and a positive divisor ``y``, return
    ``c · centropy(x, y)``; else ``None``.

    ``centropy(x, y) = x·log(x/y)`` is the GAMS relative-entropy intrinsic, which
    AMPL/GAMS lower into this product when emitting a ``.nl`` file. It is jointly
    convex on ``x ≥ 0, y > 0``, so recovering the intrinsic lets the convexity
    detector certify a Gibbs/KL objective ``Σ nᵢ·log(nᵢ/Σnⱼ)`` on the convex fast
    path (issue #207). The match is strict: exactly one bare ``x`` and one
    ``log(x/y)`` whose *numerator* is that same ``x``; ``y`` is any other factor's
    free expression. The domain guard (``x.lb ≥ 0`` finite, ``y > 0`` finite) keeps
    the substitution consistent with the entropy domain and never introduces
    ``centropy`` where the original product was outside it.
    """
    coeff = 1.0
    log_num: Expression | None = None
    log_num_idx: int | None = None
    log_den: Expression | None = None
    bare_idx: int | None = None
    bare_count = 0
    log_count = 0

    for factor in _flatten_product(expr):
        sign, factor = _strip_neg(factor)
        coeff *= sign
        if isinstance(factor, Constant) and factor.value.ndim == 0:
            coeff *= float(factor.value)
            continue
        # log(num / den)?
        if (
            isinstance(factor, FunctionCall)
            and factor.func_name == "log"
            and len(factor.args) == 1
            and isinstance(factor.args[0], BinaryOp)
            and factor.args[0].op == "/"
        ):
            num_idx = _get_flat_index(factor.args[0].left, model)
            if num_idx is not None:
                log_count += 1
                log_num = factor.args[0].left
                log_num_idx = num_idx
                log_den = factor.args[0].right
                continue
            return None
        # bare variable leaf?
        idx = _get_flat_index(factor, model)
        if idx is not None:
            bare_count += 1
            bare_idx = idx
            continue
        return None

    if (
        log_count != 1
        or bare_count != 1
        or log_num is None
        or log_den is None
        or log_num_idx != bare_idx
    ):
        return None

    # Domain guard: x nonnegative & finite (entropy domain), y strictly positive
    # & finite (log(y) and the centropy domain).
    lo_x, hi_x = _bound_expression(log_num, model)
    if not (np.isfinite(lo_x) and np.isfinite(hi_x)) or lo_x < 0.0:
        return None
    lo_y, hi_y = _bound_expression(log_den, model)
    if not (np.isfinite(lo_y) and np.isfinite(hi_y)) or lo_y <= 0.0:
        return None

    centropy = FunctionCall("centropy", log_num, log_den)
    if coeff == 1.0:
        return centropy
    return BinaryOp("*", Constant(coeff), centropy)


def _split_additive(expr: Expression) -> list[tuple[float, Expression]]:
    """Flatten a ``+``/``-``/``neg`` tree into ``[(sign, term), ...]`` leaves.

    Only additive structure is peeled — every non-additive node (variable,
    ``log``, product, ...) becomes one ``(±1.0, node)`` leaf. Used to reach the
    ``log(x)`` term hidden inside a distributed factor ``(affine + log(x))``.
    """
    terms: list[tuple[float, Expression]] = []
    stack: list[tuple[Expression, float]] = [(expr, 1.0)]
    while stack:
        node, s = stack.pop()
        if isinstance(node, BinaryOp) and node.op == "+":
            stack.append((node.left, s))
            stack.append((node.right, s))
        elif isinstance(node, BinaryOp) and node.op == "-":
            stack.append((node.left, s))
            stack.append((node.right, -s))
        elif isinstance(node, UnaryOp) and node.op == "neg":
            stack.append((node.operand, -s))
        else:
            terms.append((s, node))
    return terms


def _entropy_log_intrinsic(term: Expression, x_idx: int, model: Model) -> Expression | None:
    """If *term* is ``log(x)`` (-> ``entropy(x)``) or ``log(x/y)`` (-> ``centropy(x,
    y)``) whose entropy variable is the flat index *x_idx*, return the intrinsic
    ``FunctionCall`` (domain-guarded); else ``None``.

    The returned intrinsic stands in for ``x · term`` — the caller supplies the
    bare ``x`` factor, so ``x·log(x) = entropy(x)`` and ``x·log(x/y) =
    centropy(x, y)``.
    """
    if not (isinstance(term, FunctionCall) and term.func_name == "log" and len(term.args) == 1):
        return None
    arg = term.args[0]
    # log(x/y) -> centropy(x, y)
    if isinstance(arg, BinaryOp) and arg.op == "/":
        if _get_flat_index(arg.left, model) != x_idx:
            return None
        lo_x, hi_x = _bound_expression(arg.left, model)
        if not (np.isfinite(lo_x) and np.isfinite(hi_x)) or lo_x < 0.0:
            return None
        lo_y, hi_y = _bound_expression(arg.right, model)
        if not (np.isfinite(lo_y) and np.isfinite(hi_y)) or lo_y <= 0.0:
            return None
        return FunctionCall("centropy", arg.left, arg.right)
    # log(x) -> entropy(x)
    if _get_flat_index(arg, model) != x_idx:
        return None
    lo, hi = _bound_expression(arg, model)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0.0:
        return None
    return FunctionCall("entropy", arg)


def _match_entropy_affine_product(expr: Expression, model: Model) -> Expression | None:
    """Match the *distributed* entropy form ``c · x · (affine + log(x))`` and pull
    out the intrinsic, returning ``c·(x·affine) + c·entropy(x)`` (or ``centropy``
    when the log is ``log(x/y)``); else ``None``.

    AMPL/GAMS frequently lower ``x·log(x)`` already folded into an affine wrapper,
    e.g. ``x·(0.28809 + log(x))`` (the ``ex6_1_4`` Gibbs objective, issue #207).
    The strict :func:`_match_entropy_product` cannot see the ``log(x)`` because it
    sits inside a ``+`` factor, so the lowered product reaches the relaxer as an
    un-decomposable ``x·log(x)`` and the objective falls back to a constant floor.

    This matcher requires exactly one bare variable ``x`` and one other (non-
    constant) factor that is an additive expression containing exactly one
    ``log(x)``/``log(x/y)`` term whose entropy variable is that same ``x``. The
    affine remainder is kept as an ordinary product ``x·affine`` (linear when the
    remainder is constant, bilinear otherwise) for the existing relaxation, while
    only the entropy term is replaced. The rewrite is exact:
    ``x·(affine + log(x)) ≡ x·affine + entropy(x)``.
    """
    coeff = 1.0
    bare_idx: int | None = None
    bare_expr: Expression | None = None
    bare_count = 0
    other_factor: Expression | None = None
    other_count = 0

    for factor in _flatten_product(expr):
        sign, factor = _strip_neg(factor)
        coeff *= sign
        if isinstance(factor, Constant) and factor.value.ndim == 0:
            coeff *= float(factor.value)
            continue
        idx = _get_flat_index(factor, model)
        if idx is not None:
            bare_count += 1
            bare_idx = idx
            bare_expr = factor
            continue
        other_count += 1
        other_factor = factor

    if (
        bare_count != 1
        or other_count != 1
        or other_factor is None
        or bare_idx is None
        or bare_expr is None
    ):
        return None

    # Split the other factor additively and find the single entropy log term.
    intrinsic: Expression | None = None
    intrinsic_sign = 1.0
    rest: list[tuple[float, Expression]] = []
    for s, term in _split_additive(other_factor):
        intr = _entropy_log_intrinsic(term, bare_idx, model)
        if intr is not None:
            if intrinsic is not None:
                return None  # more than one entropy-log term -> ambiguous, bail
            intrinsic = intr
            intrinsic_sign = s
        else:
            rest.append((s, term))

    if intrinsic is None:
        return None

    # Entropy contribution: (coeff · intrinsic_sign) · entropy(x).
    intr_coeff = coeff * intrinsic_sign
    entropy_part: Expression = (
        intrinsic if intr_coeff == 1.0 else BinaryOp("*", Constant(intr_coeff), intrinsic)
    )

    if not rest:
        return entropy_part

    # Affine remainder, kept as an ordinary product coeff · x · rest.
    rest_expr: Expression | None = None
    for s, term in rest:
        node = term if s > 0 else UnaryOp("neg", term)
        rest_expr = node if rest_expr is None else BinaryOp("+", rest_expr, node)
    assert rest_expr is not None
    rest_expr = _canonicalize_entropy_expr(rest_expr, model)
    product: Expression = BinaryOp("*", bare_expr, rest_expr)
    if coeff != 1.0:
        product = BinaryOp("*", Constant(coeff), product)

    return BinaryOp("+", product, entropy_part)


def _canonicalize_entropy_expr(expr: Expression, model: Model) -> Expression:
    """Return *expr* with every ``c·x·log(x)`` product rewritten to ``c·entropy(x)``
    and every ``c·x·log(x/y)`` product rewritten to ``c·centropy(x, y)`` — including
    the distributed ``c·x·(affine + log(x))`` form (issue #207, ex6_1_4).
    Identity-preserving: unchanged subtrees keep their object identity so
    untouched models are returned byte-for-byte unchanged."""
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            matched = _match_entropy_product(expr, model)
            if matched is None:
                matched = _match_centropy_product(expr, model)
            if matched is None:
                matched = _match_entropy_affine_product(expr, model)
            if matched is not None:
                return matched
        left = _canonicalize_entropy_expr(expr.left, model)
        right = _canonicalize_entropy_expr(expr.right, model)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _canonicalize_entropy_expr(expr.operand, model)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    if isinstance(expr, FunctionCall):
        new_args = tuple(_canonicalize_entropy_expr(a, model) for a in expr.args)
        if all(n is o for n, o in zip(new_args, expr.args)):
            return expr
        return FunctionCall(expr.func_name, *new_args)
    if isinstance(expr, SumExpression):
        operand = _canonicalize_entropy_expr(expr.operand, model)
        if operand is expr.operand:
            return expr
        return SumExpression(operand, axis=expr.axis)
    if isinstance(expr, SumOverExpression):
        new_terms = [_canonicalize_entropy_expr(t, model) for t in expr.terms]
        if all(n is o for n, o in zip(new_terms, expr.terms)):
            return expr
        return SumOverExpression(new_terms)
    return expr


def canonicalize_entropy(model: Model) -> Model:
    """Return a model equivalent to *model* with entropy-family products (in the
    objective or any constraint) rewritten to their intrinsics (issue #207):

    * ``c·x·log(x)``    -> ``c·entropy(x)``
    * ``c·x·log(x/y)``  -> ``c·centropy(x, y)``   (relative entropy / Gibbs/KL)

    Both intrinsics carry dedicated relaxation / convexity support, so recovering
    them from the raw products AMPL/GAMS emit lets the bound tighten (and a
    separable entropy / relative-entropy objective is detected as convex, taking
    the convex fast path).

    The rewrites are exact (``entropy(x) ≡ x·log(x)``, ``centropy(x,y) ≡
    x·log(x/y)``) and convexity-preserving, so they run unconditionally. If
    nothing matches, *model* is returned unchanged (zero overhead, zero
    behavioural change). On any unexpected error the original model is returned,
    so the pass can never break a previously-solvable model.
    """
    try:
        changed = False

        new_objective = model._objective
        if model._objective is not None:
            new_expr = _canonicalize_entropy_expr(model._objective.expression, model)
            if new_expr is not model._objective.expression:
                from discopt.modeling.core import Objective

                new_objective = Objective(new_expr, model._objective.sense)
                changed = True

        new_constraints: list = []
        for c in model._constraints:
            if isinstance(c, Constraint):
                new_body = _canonicalize_entropy_expr(c.body, model)
                if new_body is not c.body:
                    new_constraints.append(Constraint(new_body, c.sense, c.rhs, c.name))
                    changed = True
                    continue
            new_constraints.append(c)

        if not changed:
            return model

        new_model = Model(model.name)
        new_model._variables = list(model._variables)
        new_model._parameters = list(model._parameters)
        new_model._objective = new_objective
        new_model._constraints = new_constraints
        return new_model
    except Exception:  # pragma: no cover - defensive: never break a solve
        return model


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
            # Soundness gate: clearing multiplies the constraint through by the
            # denominator, which can pull an unbounded linear term into a
            # nonlinear product (``x4 * D``) that has no valid finite envelope.
            # Such a cleared relaxation can exclude feasible points and certify a
            # false infeasibility (gear4). Keep the original quotient in that case.
            if (body is not c.body or sense != c.sense) and _has_unbounded_nonlinear_term(
                body, new_model
            ):
                rebuilt.append(c)
                continue
            if clear_only:
                # Only touch constraints the clearing actually rewrote; leave
                # everything else byte-for-byte identical so convex structure
                # elsewhere is preserved.
                if body is c.body and sense == c.sense:
                    rebuilt.append(c)
                else:
                    rebuilt.append(Constraint(distribute_products(body), sense, c.rhs, c.name))
                continue
            body = _prelift_blowup_products(body, new_model, lifter)
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
            obj_expr = _prelift_blowup_products(
                new_model._objective.expression, new_model, lifter
            )
            obj_expr = distribute_products(obj_expr)
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
