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

from typing import Callable, Optional, TypeVar

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    carry_validation_guards,
)
from discopt.mpec import carry_complementarities

from .gdp_reformulate import (
    _bound_expression,
    _BoundMemo,
    _collect_variables,
    _is_linear,
    bound_expression_error,
)
from .term_classifier import (
    _affine_atom_key,
    _affine_walk,
    _get_flat_index,
    distribute_products,
    distribution_exceeds_budget,
)
from .term_classifier import estimate_distributed_terms as _estimate_distributed_terms

# A denominator counts as sign-definite only when its interval is bounded away
# from zero by at least this margin — guards against a denominator that merely
# grazes zero (where the multiply-through would not be value-preserving).
_ZERO_MARGIN = 1e-9
# Reject an aux lift whose induced bound magnitude is effectively infinite: an
# unbounded monomial aux would make the relaxation/NLP unbounded.
_INF_THRESH = 1e15


def _lift_zero_spanning_factors_enabled() -> bool:
    """R4 feature flag (``DISCOPT_LIFT_ZERO_SPANNING_FACTORS``, default **ON** since
    G1.5; ``DISCOPT_LIFT_ZERO_SPANNING_FACTORS=0`` is the escape hatch).

    When a product ``f(x)·g(x)`` has a *non-atomic* factor ``f`` whose interval
    spans 0, the factorable reform already lifts ``w == f`` to a bounded aux (via
    :func:`_prelift_blowup_products`). But the default spatial-branching policy
    then *deprioritizes* every lifted aux column (``solver.py`` — a product aux
    ``w = x_i·x_j`` cannot shrink its own envelope, so bisecting it is wasteful).
    For a zero-*spanning* factor that reasoning is inverted: branching ``w`` at 0
    splits the sign of the factor, and the McCormick envelope of the product
    ``w·g`` (with ``g ≥ 0``) responds sharply — it is the only move that un-pins
    the bound (st_e36: root −304.5, pinned for every x-box, jumps to ≈ optimum
    once ``w`` is split at 0; see uncertified-tail-plan §3 R4). This flag marks
    those specific auxes so the solver keeps them branchable.

    Structure-gated: where no product has a zero-spanning non-atomic factor, no aux
    is tagged and the reform is byte-identical (inert) — so this flag is a free win
    on-structure (st_e36 feasible→optimal) and invisible elsewhere. Graduated to
    default-ON (G1.5-redo, post-C-38) on gate evidence: the isolated held-out arm
    (N=40, seed 0, tl 25 s) verdicts eligible — 0 soundness violations, cert-neutral
    (bound-changing regime), regression 3.8 % (1 off-structure timing artifact, ≪
    the 10 % ceiling) — plus the bound-changing verification (differential dual
    bound tighter but still ≤ =opt= on st_e36: −304.5 → −246.02 ≤ −246.0;
    feasible-point sample recovers the identical incumbent ON vs OFF). See
    ``docs/dev/flag-graduation-redo-2026-07-07.md``. ``=0`` restores the old
    byte-identical (no-tagging) behavior.
    """
    import os

    return os.environ.get("DISCOPT_LIFT_ZERO_SPANNING_FACTORS", "1") != "0"


def _lift_loose_products_enabled() -> bool:
    """TD-A feature flag (``DISCOPT_LIFT_LOOSE_PRODUCTS``, default ON).

    Graduated per T2.6 with 3 consecutive green held-out verdicts (composed
    with the density LU route): BR-3 #602 (verdict 1), FLAG-GRAD #612
    (verdict 2), and the P0 SPATIAL-CERT re-run
    (``docs/dev/p0-spatial-cert-2026-07-10.md``, verdict 3 — incorrect 0,
    oracle-cross 0, cert-loss 0; nvs09 dual bound −63.71 -> −50.27). Set
    ``DISCOPT_LIFT_LOOSE_PRODUCTS=0`` to restore the old default.

    Extends the factorable lift to an integer power of a *non-atomic univariate
    function call* — ``g(x)**n`` with ``g`` a transcendental (``log``, ``sin``,
    …) whose argument is not a bare variable and ``n`` a positive integer ``≥ 2``.

    The MILP objective/constraint linearizer bounds a power only when its base is
    a bare variable index (the monomial path) or when the power is a *fractional*
    power of a lifted composite (``_lift_objective_atoms``). An *integer* power of
    a call falls through both: ``distribute_products`` expands ``g(x)**2`` into the
    product ``g(x)·g(x)``, which ``_decompose_product`` cannot decompose (both
    factors are transcendental), so the whole term is dropped and the model loses
    its dual bound (nvs09: ``Cannot decompose product: log·log`` → feasibility
    objective, 69 % root gap; mathopt5_6: ``sin·sin`` → no bound at all).

    The lift introduces ``t == g(x)`` (a bounded aux via :meth:`_Lifter.expression`
    — ``t`` inherits ``g``'s FBBT interval, e.g. ``log(x-2) ∈ [0, log 7]`` for
    ``x ∈ [3, 9]``) and rewrites the node as the *monomial* ``t**n``, which the
    existing monomial-secant / even-power envelope relaxes exactly (even ``n``) or
    3-regime (odd ``n``). This is an exact identity substitution (``t == g(x)``),
    so it cuts no feasible point; it only replaces a *dropped* term with a
    relaxable one. ``DISCOPT_LIFT_LOOSE_PRODUCTS=0`` keeps the reform
    byte-identical to the pre-lift behavior.

    Entry-experiment measurement (nvs09 hand-lift): root bound −72.90 → −54.83
    (root gap 69.0 % → 27.1 %, a 60.7 % relative reduction, ≥ 25 % bar). nvs05 was
    killed separately — its objective monomial is already exactly relaxed (the
    box-minimum 0.674 equals the root bound), so its gap is constraint-driven box
    reduction (OBBT/branch-and-reduce), not a lifting problem. See
    ``docs/dev/uncertified-tail-plan-results-2026-07-06.md`` §TD-A.
    """
    import os

    return os.environ.get("DISCOPT_LIFT_LOOSE_PRODUCTS", "1") != "0"


# Transcendental univariate calls whose integer power TD-A lifts. Restricted to
# single-argument functions with a monotone/bounded envelope so ``t == g(x)``
# always yields a finite aux interval; excludes ``abs``/``sign`` (non-smooth) and
# n-ary ``min``/``max``/``prod``/``norm`` (not univariate).
_LIFTABLE_CALL_POWER_FUNCS = frozenset(
    {"log", "log2", "log10", "exp", "sqrt", "sin", "cos", "tan", "atan", "tanh", "log1p"}
)


def _liftable_call_power_base(expr: Expression, model: Model) -> tuple[FunctionCall, int] | None:
    """If *expr* is ``g(x)**n`` with ``g`` a single-argument transcendental over a
    *non-atomic* argument (not a bare variable) and ``n`` a positive integer
    ``≥ 2``, return ``(g_call, n)``; otherwise ``None``.

    A bare-variable base (``x**n``) is already the monomial path and must not be
    lifted; a fractional power is handled by ``_lift_objective_atoms`` directly.
    """
    if not isinstance(expr, BinaryOp) or expr.op != "**":
        return None
    if not isinstance(expr.right, Constant):
        return None
    exp_val = float(expr.right.value)
    n = int(exp_val)
    if exp_val != n or n < 2:
        return None
    base = expr.left
    if not isinstance(base, FunctionCall):
        return None
    if base.func_name not in _LIFTABLE_CALL_POWER_FUNCS or len(base.args) != 1:
        return None
    # A univariate call over a bare variable (``sin(x)**2``) is still worth
    # lifting: the linearizer drops it exactly the same way (the base is a call,
    # not a variable index). Only require that the call is not itself trivially a
    # constant.
    return base, n


# The factorable-reform walkers (``_find_clearable_denominator``, ``_lift_expr``,
# the ``has_factorable_work`` scanners, ...) recurse one Python frame per
# expression node.  ``from_nl`` rebuilds a sum/product of N terms as a left-deep
# binary tree of depth ~N, so a *single* constraint body can be tens of
# thousands of nodes deep (watercontamination0202r: ~53k, graphpart_clique-30:
# ~7.6k) and overrun CPython's default 1000-frame recursion limit, raising an
# *uncaught* ``RecursionError`` (issue #271).  Mirroring the convexity walk's fix
# (issue #266), the public entry points run the walk with a depth-scaled
# recursion limit on a large-stack worker thread; this is gated on the deepest
# expression so the common (shallow) case still runs inline with the default
# limit, leaving behaviour byte-for-byte identical there.
#
# NOTE: the gate is driven by the *maximum single-expression node count*, not the
# constraint/variable count used by the convexity walk's gate — the hazard here
# is depth concentrated in one giant body, which a constraint count misses
# entirely (this model has only ~280 constraints but a 53k-node body).
_DEEP_RECURSION_SIZE_GATE = 700


def _expr_node_count_children(node: Expression) -> tuple[Expression, ...]:
    if isinstance(node, BinaryOp):
        return (node.left, node.right)
    if isinstance(node, UnaryOp):
        return (node.operand,)
    if isinstance(node, FunctionCall):
        return tuple(node.args)
    if isinstance(node, SumExpression):
        return (node.operand,)
    if isinstance(node, SumOverExpression):
        return tuple(node.terms)
    return ()


def _expr_node_count(expr: Expression) -> int:
    """Count the nodes in *expr* iteratively (so this measurement never itself
    recurses into the very depth it is trying to size).

    The count is of the expression *as a tree* (a shared subexpression counts
    once per occurrence), which is what the recursion-headroom gates were sized
    against. #1565: it is computed bottom-up over the DAG -- ``size(n) = 1 +
    sum(size(child))``, memoised by node identity -- instead of by visiting every
    occurrence, which was exponential on a reduced-space NN embedding. Same
    number, linear time.
    """
    sizes: dict[int, int] = {}
    held: list[Expression] = []  # keep every sized node alive: ids cannot recycle
    stack: list[tuple[Expression, bool]] = [(expr, False)]
    while stack:
        node, expanded = stack.pop()
        nid = id(node)
        if nid in sizes:
            continue
        children = _expr_node_count_children(node)
        if not expanded and children:
            stack.append((node, True))
            stack.extend((c, False) for c in children if id(c) not in sizes)
            continue
        sizes[nid] = 1 + sum(sizes[id(c)] for c in children)
        held.append(node)
    return sizes[id(expr)]


def _max_expr_node_count(model: Model) -> int:
    """Largest single-expression node count across the objective and constraint
    bodies — an upper bound on how deep the per-expression walk can recurse."""
    largest = 0
    obj = getattr(model, "_objective", None)
    if obj is not None:
        largest = _expr_node_count(obj.expression)
    for c in getattr(model, "_constraints", []) or []:
        if isinstance(c, Constraint):
            largest = max(largest, _expr_node_count(c.body))
    return largest


def _recursion_headroom_need(model: Model) -> int:
    """Estimate the recursion-limit headroom the factorable walk may need.

    Returns 0 when the deepest expression is small enough that the default limit
    is safe. The estimate is a deliberate over-approximation (recursion depth is
    bounded by node count); it only affects whether the deep-stack path engages,
    never what the walk detects.
    """
    # #1520: no except. ``_max_expr_node_count`` is an iterative walk over the
    # model's own expressions with no failure mode; the old handler could only turn
    # a defect into "no headroom" and a later unexplained RecursionError.
    size = _max_expr_node_count(model)
    if size <= _DEEP_RECURSION_SIZE_GATE:
        return 0
    # A few Python frames are entered per expression node along the deepest path;
    # add a fixed cushion for the surrounding call stack.  Capped (matching the
    # convexity walk's, issue #266) so a pathological size can't request an
    # unsatisfiable limit.
    return min(2000 + 6 * size, 600_000)


_T = TypeVar("_T")


def _run_factorable_with_headroom(model: Model, fn: Callable[[], _T]) -> _T:
    """Run ``fn`` (a factorable-reform walk over *model*) with size-scaled
    recursion headroom so deep expression graphs don't ``RecursionError``.

    Delegates to the proven worker-thread runner from the convexity module
    (issue #266); shallow models run ``fn`` inline at the default limit.
    """
    from .convexity.rules import _run_with_deep_recursion

    return _run_with_deep_recursion(fn, depth_need=_recursion_headroom_need(model))


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


def _structural_key(expr: Expression, pins: list) -> tuple:
    """An exact, hashable structural key for *expr* (#1555).

    Two expressions get the same key only if they are the same expression:

    * constants by dtype, shape and raw bytes (``repr`` rounds to ``.6g`` and
      prints arrays by shape, #1497);
    * variables and parameters by identity -- the model keeps them alive, so
      their ``id()`` cannot be recycled during the lift (the ex7_2_3 hazard);
    * an ``IndexExpression``'s index exactly (an ndarray index by its bytes);
    * any node type not listed below by ``id()``, with the node appended to
      *pins* so its address stays reserved for as long as the cache lives.

    Never contains an expression object itself: ``Expression.__eq__`` builds a
    ``Constraint``, which would break dict lookup on a hash collision.
    """
    memo: dict[int, tuple] = {}

    def index_key(idx) -> tuple:
        if isinstance(idx, np.ndarray):
            return ("nd", idx.dtype.str, idx.shape, idx.tobytes())
        if isinstance(idx, tuple):
            return ("t",) + tuple(index_key(i) for i in idx)
        if isinstance(idx, list):  # elementwise: repr truncates large nested arrays
            return ("l",) + tuple(index_key(i) for i in idx)
        if isinstance(idx, slice):
            return ("s", idx.start, idx.stop, idx.step)
        if isinstance(idx, (bool, np.bool_)):  # before int: bool subclasses int
            return ("b", bool(idx))
        if isinstance(idx, (int, np.integer)):
            return ("i", int(idx))
        return ("r", type(idx).__name__, repr(idx))  # Ellipsis / None: exact reprs

    def walk(node: Expression) -> tuple:
        nid = id(node)
        hit = memo.get(nid)
        if hit is not None:
            return hit
        if isinstance(node, Constant):
            v = np.asarray(node.value)
            k: tuple = ("C", v.dtype.str, v.shape, v.tobytes())
        elif isinstance(node, (Variable, Parameter)):
            k = (type(node).__name__, id(node))
        elif isinstance(node, IndexExpression):
            k = ("I", walk(node.base), index_key(node.index))
        elif isinstance(node, BinaryOp):
            k = ("B", node.op, walk(node.left), walk(node.right))
        elif isinstance(node, UnaryOp):
            k = ("U", node.op, walk(node.operand))
        elif type(node) is FunctionCall:
            k = ("F", node.func_name) + tuple(walk(a) for a in node.args)
        elif isinstance(node, SumOverExpression):
            k = ("SO",) + tuple(walk(t) for t in node.terms)
        elif isinstance(node, SumExpression):
            k = ("S", node.axis, walk(node.operand))
        elif isinstance(node, MatMulExpression):
            k = ("M", walk(node.left), walk(node.right))
        else:
            pins.append(node)
            k = ("id", type(node).__name__, nid)
        memo[nid] = k
        return k

    return walk(expr)


class _LiftAbandoned(Exception):
    """The solve clock ran out inside one constraint's lift (#1565).

    Private to this module: raised only by :meth:`_Lifter.tick` and caught only
    by :func:`_factorable_reformulate_inner`, which then discards everything the
    lifter built and returns the model it was given -- the same wholesale
    abandonment the per-constraint #1456 check performs, reached from deeper in
    the traversal. It is never a defect signal, so catching it hides nothing.
    """


# How many lift-walker visits pass between two consultations of the deadline.
# The walkers are cheap per visit; the expensive steps (a convexity
# classification, an aux bound) consult it unconditionally.
_LIFT_DEADLINE_STRIDE = 256


class _Lifter:
    """Allocates monomial auxiliary variables ``w == leaf**k`` on *model*,
    deduplicating by (flat_index, exponent)."""

    def __init__(self, model: Model, deadline: Optional[Callable[[], bool]] = None):
        self.model = model
        # #1565: the #1456 deadline, consulted *inside* a constraint's lift. A
        # single reduced-space NN constraint can hold the whole network, so a
        # per-constraint check alone cannot stop a lift that never reaches the
        # next constraint.
        self._deadline = deadline
        self._ticks = 0
        # #1565: ``_should_lift_call_arg`` per call node, keyed by ``id`` and
        # holding the node so the id cannot be recycled while the lifter lives.
        # The verdict reads only the node and the bounds of the variables in it,
        # neither of which the lift changes, so a memo hit returns exactly what
        # a recomputation would.
        self._call_arg_memo: dict[int, tuple[Expression, Expression | None]] = {}
        # #1565: ``_scan_for_liftable_call_power`` verdicts (pure, node-held).
        self.call_power_memo: dict[int, tuple[Expression, bool]] = {}
        self._cache: dict[tuple[int, int], Variable] = {}
        # Keyed by an EXACT structural key (:func:`_structural_key`), never a bare
        # ``id()`` of an expression node and never ``repr()``:
        #
        # * ``id()``: CPython recycles the ``id()`` of a garbage-collected object,
        #   so a later, structurally *different* expression can reuse a freed
        #   address and score a false cache hit. In ex7_2_3 that dropped a ``/x8``
        #   denominator from a lifted ratio and certified an infeasible box corner
        #   as the global optimum (a false "optimal").
        # * ``repr()``: a DISPLAY string, and lossy -- ``SumOverExpression`` prints
        #   as ``"Σ[n terms]"``, so ``dm.sum([x, 1])`` and ``dm.sum([y, 1])`` shared
        #   one aux and ``(y+1)**1.7`` was silently rewritten as ``(x+1)**1.7``: a
        #   certified 1.8095 against a true 4.0 (#1555).
        #
        # Only a LOSSLESS key can fail solely by not deduplicating (an extra aux,
        # harmless) and never by merging two distinct expressions.
        self._expr_cache: dict[tuple, Variable] = {}
        # Nodes whose key falls back to ``id()`` are pinned here, so their address
        # cannot be recycled while this lifter (and its cache) is alive.
        self._key_pins: list[Expression] = []
        self.aux_constraints: list[Constraint] = []
        self._counter = 0
        # R4: names of lifted product-factor auxes whose interval spans 0. These
        # are the only auxes worth keeping as spatial-branching candidates (see
        # ``_lift_zero_spanning_factors_enabled``). Populated only when the flag
        # is on, so the default reform is byte-identical.
        self.zero_spanning_factor_auxes: set[str] = set()
        # Auxes created with ``integer=True`` (an exact integer-valued affine
        # definition): integral because of the columns they are defined by, see
        # ``Model._implied_integer_auxes``. Every such aux, whichever rule made it
        # (the translated-monomial lift or main's #1544 cancellation / term-limit
        # path). Populated only with ``DISCOPT_LIFT_AFFINE_MONOMIALS`` on (the
        # default), so the ``=0`` opt-out reproduces pre-#1588 main exactly. The
        # default-path handling has its own panel (#1593): see
        # ``primal_heuristics._is_free_integer``.
        self.implied_integer_auxes: set[str] = set()

    def tick(self, *, force: bool = False) -> None:
        """Charge one walker visit; raise :class:`_LiftAbandoned` once the clock
        is spent. ``force`` consults the deadline now (before an expensive step)
        rather than at the next stride boundary."""
        if self._deadline is None:
            return
        self._ticks += 1
        if (force or self._ticks % _LIFT_DEADLINE_STRIDE == 0) and self._deadline():
            raise _LiftAbandoned(f"time limit spent after {self._ticks} lift visits")

    def call_arg_to_lift(self, call: Expression) -> Expression | None:
        """Memoised :func:`_should_lift_call_arg` (see ``_call_arg_memo``)."""
        hit = self._call_arg_memo.get(id(call))
        if hit is not None:
            return hit[1]
        self.tick(force=True)
        arg = _should_lift_call_arg(call, self.model)
        self._call_arg_memo[id(call)] = (call, arg)
        return arg

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
        if lo > hi:
            # Never seed an inverted aux box (matches ``.expression``); an empty
            # interval would make the nonlinear bound-tightening wrongly prove
            # infeasibility.
            return None
        name = f"_fr_aux_{self._counter}"
        self._counter += 1
        w = Variable(name, VarType.CONTINUOUS, (), lo, hi, self.model)
        self.model._variables.append(w)
        self._cache[key] = w
        # Normalised form ``body == 0`` with body = w - leaf**exp.
        self.aux_constraints.append(Constraint(BinaryOp("-", w, pow_expr), "==", 0.0))
        return w

    def expression(
        self, expr: Expression, *, lb_floor: float | None = None, integer: bool = False
    ) -> Variable | None:
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

        *integer*, when true, types a NEWLY created aux ``INTEGER``. The caller
        asserts *expr* is integer-valued at every integer-feasible point
        (:func:`_is_integer_valued_affine`), so the type is exact, never a
        restriction. A cached aux is returned as it was created.
        """
        self.tick(force=True)
        key = (_structural_key(expr, self._key_pins), lb_floor)
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
        vtype = VarType.INTEGER if integer else VarType.CONTINUOUS
        w = Variable(name, vtype, (), lo, hi, self.model)
        if integer and _lift_affine_monomials_enabled():
            self.implied_integer_auxes.add(name)
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
        Requires ``base_var >= 0`` so the power is real and monotone (increasing
        for ``p > 0``, decreasing for ``p < 0``); the induced box is taken over the
        endpoints in either case.
        """
        lo = float(np.min(base_var.lb))
        hi = float(np.max(base_var.ub))
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo < 0.0:
            return None
        d_lo, d_hi = lo**p, hi**p
        # ``base_var >= 0`` makes ``base**p`` monotone on ``[lo, hi]``, so the
        # interval extremes are at the endpoints — but for a NEGATIVE power it is
        # *decreasing* (``lo**p > hi**p``), so the raw ``(lo**p, hi**p)`` is
        # inverted. Order the endpoints. An unordered (inverted) pair would seed an
        # aux box with ``lb > ub`` that the nonlinear bound-tightening reads as an
        # empty interval and wrongly proves the model infeasible (hda: known
        # optimum -5964.5 was returned as `infeasible`).
        if d_lo > d_hi:
            d_lo, d_hi = d_hi, d_lo
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

# A product is also lifted, whatever its term count, when multiplying it out would
# CANCEL catastrophically (#1544). Expanding ``prod_k f_k`` sums terms whose
# magnitudes multiply ``mag(f_k) = sum_j max|t_kj|`` over the box, while the value
# itself is at most ``prod_k max|f_k|``; the ratio ``prod_k mag(f_k) / max|f_k|``
# is the factor by which float64 rounding in the expanded polynomial is amplified.
# Under an exact change of variables ``x = y - c`` it grows like
# ``(2c / width)**n``: nvs09 shifted by c ~ 1e3 has ten factors ``y_k - c_k`` over
# width-6 boxes, ratio ~1e25, and its distributed objective evaluated to -1123.3
# at a point whose true value is 10.87. At this limit the amplified rounding is
# ~2e-10 relative, below the solver's tolerances; above it the factors are lifted
# to auxes ``w_k == f_k`` (exact equalities, each well conditioned).
_DISTRIBUTE_CANCELLATION_LIMIT = 1e6


def _additive_terms(expr: Expression) -> list[Expression]:
    """Flatten ``+``/``-``/``neg`` into additive terms (signs dropped)."""
    if isinstance(expr, BinaryOp) and expr.op in ("+", "-"):
        return _additive_terms(expr.left) + _additive_terms(expr.right)
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _additive_terms(expr.operand)
    return [expr]


def _distribution_cancellation(factors: list[Expression], model: Model) -> float:
    """Rounding amplification of multiplying *factors* out (see
    :data:`_DISTRIBUTE_CANCELLATION_LIMIT`).

    A factor whose magnitude cannot be measured (an unbounded term, or a factor
    identically zero over the box) contributes 1: "cannot measure" is not evidence
    of cancellation, and such a factor could not be lifted to a bounded aux anyway.
    """
    ratio = 1.0
    for f in factors:
        terms = _additive_terms(f)
        if len(terms) < 2:
            continue
        mag = 0.0
        for t in terms:
            t_lo, t_hi = _bound_expression(t, model)
            mag += max(abs(t_lo), abs(t_hi))
        lo, hi = _bound_expression(f, model)
        size = max(abs(lo), abs(hi))
        if not (np.isfinite(mag) and np.isfinite(size)) or size == 0.0:
            continue
        ratio *= max(1.0, mag / size)
    return ratio


def _is_integer_valued_affine(expr: Expression) -> bool:
    """True when *expr* is affine with integral coefficients and constant over
    integer/binary scalar variables, so it is integer at every integer-feasible
    point (e.g. ``y - 1450`` with ``y`` integer)."""
    if isinstance(expr, Constant):
        v = np.asarray(expr.value, dtype=np.float64)
        return v.size == 1 and bool(np.isfinite(v).all()) and float(v.reshape(())).is_integer()
    if isinstance(expr, Variable):
        return expr.var_type != VarType.CONTINUOUS and int(np.prod(expr.shape)) == 1
    if isinstance(expr, IndexExpression):
        return (
            isinstance(expr.base, Variable)
            and expr.base.var_type != VarType.CONTINUOUS
            and np.ndim(np.empty(expr.base.shape)[expr.index]) == 0
        )
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _is_integer_valued_affine(expr.operand)
    if isinstance(expr, SumOverExpression) and _lift_affine_monomials_enabled():
        # Recentring writes ``y + c`` this way (#1537). Gated with the lift that
        # introduced it, so ``=0`` is the pre-#1537 rule exactly.
        return bool(expr.terms) and all(_is_integer_valued_affine(t) for t in expr.terms)
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            return _is_integer_valued_affine(expr.left) and _is_integer_valued_affine(expr.right)
        if expr.op == "*":
            for a, b in ((expr.left, expr.right), (expr.right, expr.left)):
                if isinstance(a, Constant) and _is_integer_valued_affine(a):
                    return _is_integer_valued_affine(b)
    return False


def _collect_mul_factors(expr: Expression) -> list[Expression]:
    """Flatten a left/right-nested ``*`` chain into its factor list.

    Left-to-right order, iterative and linear in the chain length (#1456): the
    recursive ``left + right`` list concatenation was quadratic in it, and the
    walkers that call this at every ``*`` node of a chain made it cubic.
    """
    out: list[Expression] = []
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, BinaryOp) and node.op == "*":
            stack.append(node.right)
            stack.append(node.left)
        else:
            out.append(node)
    return out


def _lift_affine_monomials_enabled() -> bool:
    """``DISCOPT_LIFT_AFFINE_MONOMIALS`` (#1537): lift the translated factors of a
    multilinear monomial instead of multiplying them out. See
    :func:`_is_translated_monomial` for the rule and its measurement.

    Default ON since its graduation panel (2026-10-02, on top of the #1586 OBBT
    cascade fix; 206 interleaved comparisons over the in-repo corpus as written,
    under 1e3/1e6 translations and the generated families;
    ``recentre_graduation_panel.py --flag``): 0 false, 0 lost, 0 neutrality drift;
    certificates 166 -> 168 (nvs05 as written, nvs01 at the 1e3 shift). Re-run
    2026-10-03 after the #1588 review fixes (maximal-chain rule, reach, implied-
    integer auxes): same verdict, certificates 167 -> 169. ``=0`` restores the
    distribute-then-cap path."""
    import os

    return os.environ.get("DISCOPT_LIFT_AFFINE_MONOMIALS", "1") != "0"


def _univariate_affine_key(expr: Expression) -> Optional[tuple[tuple, bool]]:
    """``(variable key, has_offset)`` when *expr* is ``a * v + b`` with ``v`` ONE
    scalar variable (or a scalar element of one) and ``a != 0``, else ``None``.

    ``has_offset`` is ``b != 0`` -- the factor is a translated copy of ``v``, which
    distributes into two terms. Coefficients are folded exactly (``Fraction``).
    """
    coef: dict[tuple, object] = {}
    const = [0]
    ok = [True]

    def visit(kind: str, node: Expression, scale) -> None:
        if kind == "const":
            const[0] = const[0] + scale
            return
        if isinstance(node, SumOverExpression):  # recentring writes ``y + c`` this way
            for t in node.terms:
                _affine_walk(t, None, lambda k, n, s: visit(k, n, s * scale))
            return
        key = _affine_atom_key(node)
        if key[0] == "n" or (isinstance(node, Variable) and int(np.prod(node.shape)) != 1):
            ok[0] = False
            return
        coef[key] = coef.get(key, 0) + scale

    _affine_walk(expr, None, visit)
    if not ok[0]:
        return None
    live = [k for k, a in coef.items() if a != 0]
    if len(live) != 1:
        return None
    return live[0], const[0] != 0


def _is_translated_monomial(factors: list[Expression]) -> bool:
    """True when the product of *factors* is a multilinear monomial written in
    translated coordinates: every factor is a constant or ``a_k v_k + b_k`` over
    pairwise-distinct scalar variables, at least three factors are non-constant,
    and at least one carries an offset ``b_k != 0``.

    #1537: such a product is ``prod_k (a_k v_k + b_k)``. Multiplying it out gives
    ``2**n`` multilinear terms whose term-wise relaxation depends on the offsets;
    lifting each translated factor to an exact aux ``w_k == a_k v_k + b_k`` gives
    the monomial ``prod_k w_k``, whose relaxation is the one the model would get in
    the coordinates where the offsets are zero. So the relaxation no longer depends
    on where the user put the origin. Measured on nvs09 (``- (prod_k x_k)**0.2``,
    ten integers on [3, 9]) moved by ``x = y - 3`` / ``x = y + 3``: the 1024-term
    expansion sits exactly AT :data:`_DISTRIBUTE_TERM_LIMIT` (so was not lifted)
    and its McCormick LP (10,177 x 42,518) exceeded the dense cap, leaving
    interval/alphaBB bounds of -48.0 / -81.0 and no certificate in 30 s; lifted,
    both certify -43.1343 in 39 nodes (unshifted: 31).

    Bilinear products are excluded: McCormick is exact under translation of
    either factor, so lifting changes nothing there. A repeated variable is
    excluded: ``(x - 1)(x - 2)(x - 3)`` is a univariate polynomial, which the
    expanded form relaxes better than three independent auxes would.

    Reach (#1588 review): the rule is applied to maximal ``*`` chains found by the
    prelift's walk -- through ``+``/``-``, unary nodes and ``dm.sum`` terms, and a
    product found this way counts as factorable work on its own
    (:func:`_scan_for_translated_monomial`). Not reached: a product inside a call
    argument (``exp((x-1)(y-1)(z-1))``; the call-argument lift handles the call),
    and ``dm.prod(X - 3)`` over an ARRAY, which is one ``prod`` reduction node
    that is never distributed, so the ``2**n``-term expansion this rule prevents
    does not arise for it (``dm.prod([...])`` over a list builds a ``*`` chain and
    is lifted).
    """
    seen: set[tuple] = set()
    nonconst = 0
    offset = False
    for f in factors:
        if isinstance(f, Constant):
            continue
        hit = _univariate_affine_key(f)
        if hit is None:
            return False
        key, has_offset = hit
        if key in seen:
            return False
        seen.add(key)
        nonconst += 1
        offset = offset or has_offset
    return nonconst >= 3 and offset


def _scan_for_translated_monomial(expr: Expression, *, _in_chain: bool = False) -> bool:
    """True if *expr* holds a maximal ``*`` chain that :func:`_is_translated_monomial`
    accepts, found by exactly the walk :func:`_prelift_blowup_products` makes
    (``BinaryOp`` / ``UnaryOp`` nodes, each maximal chain tested once).

    #1588 review: this is what makes a translated monomial count as factorable
    work on its own. Without it ``min (x-3)(y-1)(z-2)(u-4)`` was returned
    unchanged, because the lift only ran when some *other* lift opened the pass.
    """
    if isinstance(expr, BinaryOp):
        if expr.op == "*" and not _in_chain:
            if _is_translated_monomial(_collect_mul_factors(expr)):
                return True
        sub = expr.op == "*"
        return _scan_for_translated_monomial(
            expr.left, _in_chain=sub
        ) or _scan_for_translated_monomial(expr.right, _in_chain=sub)
    if isinstance(expr, UnaryOp):
        return _scan_for_translated_monomial(expr.operand)
    if isinstance(expr, SumOverExpression):  # ``dm.sum([...])`` (see the prelift)
        return any(_scan_for_translated_monomial(t) for t in expr.terms)
    return False


def _prelift_blowup_products(
    expr: Expression, model: Model, lifter: "_Lifter", *, _in_chain: bool = False
) -> Expression:
    """Lift the factors of any product whose naive distribution would explode.

    Walks *expr*; at each ``*``-rooted product whose estimated distributed term
    count exceeds :data:`_DISTRIBUTE_TERM_LIMIT`, or whose expansion would cancel
    catastrophically (:data:`_DISTRIBUTE_CANCELLATION_LIMIT`), lifts every multi-term-sum
    factor ``f`` to an aux ``w == f`` (an exact equality) and rebuilds the
    product over the auxes, so the relaxation sees a bilinear/multilinear product
    of bounded variables instead of a high-degree expanded polynomial.  Sound:
    each ``w == f`` is exact and the lifted product is McCormick-relaxable.
    A ``*`` node that passes none of these tests is returned as the same
    object when nothing beneath it is lifted; the pass as a whole still returns
    new ``BinaryOp`` parents above any lifted node. A lifted factor that is
    integer-valued (``y - c`` with ``y`` integer, ``c`` integral) gets an
    ``INTEGER`` aux, so the lift keeps the integrality the factor had.

    With ``DISCOPT_LIFT_AFFINE_MONOMIALS`` on, a product that
    :func:`_is_translated_monomial` accepts is lifted too. That rule is applied
    to the MAXIMAL ``*`` chain only (``_in_chain`` marks the inner ``*`` nodes of
    a chain already tested): re-testing every sub-chain made the answer depend
    on how the product was parenthesised -- ``(y-1)(z-2)(u-3)(y-4)`` lifted three
    factors while ``(y-4)(y-1)(z-2)(u-3)`` lifted none (#1588 review).
    """
    lifter.tick()
    if isinstance(expr, BinaryOp):
        translated = (
            expr.op == "*"
            and not _in_chain
            and _lift_affine_monomials_enabled()
            and _is_translated_monomial(_collect_mul_factors(expr))
        )
        if expr.op == "*" and (
            translated
            or _estimate_distributed_terms(expr) > _DISTRIBUTE_TERM_LIMIT
            or _distribution_cancellation(_collect_mul_factors(expr), model)
            > _DISTRIBUTE_CANCELLATION_LIMIT
        ):
            new_factors: list[Expression] = []
            changed = False
            for f in _collect_mul_factors(expr):
                # Only a genuine multi-term sum drives the blowup; a constant,
                # variable, or monomial power distributes to one term and is left
                # for the normal monomial/bilinear path. A translated monomial's
                # offset factors are lifted whatever node type spells them.
                if (
                    isinstance(f, BinaryOp)
                    and f.op in ("+", "-")
                    and _estimate_distributed_terms(f) >= 2
                ) or (
                    translated
                    and not isinstance(f, Constant)
                    and (_univariate_affine_key(f) or (None, False))[1]
                ):
                    # Recurse first so a factor that is *itself* a blowup product
                    # has its inner factors lifted before this one is bounded.
                    f_lifted = _prelift_blowup_products(f, model, lifter)
                    w = lifter.expression(f_lifted, integer=_is_integer_valued_affine(f_lifted))
                    if w is not None:
                        new_factors.append(w)
                        changed = True
                        # R4: a *product factor* whose lifted aux interval spans 0
                        # is the branch-responsive one (splitting w at 0 flips the
                        # factor's sign, tightening the product envelope). Tag it so
                        # the solver keeps it a spatial-branching candidate instead
                        # of deprioritizing it with the pure-product auxes. Flag-
                        # gated: no tagging when off, so the reform is unchanged.
                        if _lift_zero_spanning_factors_enabled():
                            w_lo = float(np.min(w.lb))
                            w_hi = float(np.max(w.ub))
                            if w_lo < 0.0 < w_hi:
                                lifter.zero_spanning_factor_auxes.add(w.name)
                        continue
                    new_factors.append(f_lifted)
                else:
                    new_factors.append(_prelift_blowup_products(f, model, lifter))
            if changed:
                return _rebuild_product(1.0, new_factors)
            # Couldn't bound any factor (unbounded box): leave the product to the
            # existing distribute/monomial path rather than alter it.
            return expr
        sub = expr.op == "*"
        left = _prelift_blowup_products(expr.left, model, lifter, _in_chain=sub)
        right = _prelift_blowup_products(expr.right, model, lifter, _in_chain=sub)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _prelift_blowup_products(expr.operand, model, lifter)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    if isinstance(expr, SumOverExpression) and _lift_affine_monomials_enabled():
        # ``dm.sum([...])`` is a sum node like ``+``; without this descent a
        # translated monomial written inside one was never lifted (#1588 review,
        # reach). Flag-gated with the lift so ``=0`` keeps the pre-#1537 walk,
        # which stopped here for the blowup/cancellation lifts too.
        terms = [_prelift_blowup_products(t, model, lifter) for t in expr.terms]
        if all(a is b for a, b in zip(terms, expr.terms)):
            return expr
        return SumOverExpression(terms)
    return expr


def _prelift_call_powers(expr: Expression, model: Model, lifter: "_Lifter") -> Expression:
    """TD-A pre-pass: lift an integer power of a univariate call ``g(x)**n`` (n >= 2)
    to the monomial ``t**n`` with ``t == g(x)`` a bounded aux, *before*
    ``distribute_products`` expands ``g(x)**2`` into the ``g(x)·g(x)`` product the
    downstream ``_decompose_product`` cannot linearize (so the term is dropped and
    the model loses its dual bound — nvs09 ``log·log``, mathopt5_6 ``sin·sin``).

    Runs bottom-up so a call whose argument itself contains a power-of-call is
    lifted inside out. Identity-preserving when nothing matches (returns the same
    object), so a model without this structure is byte-for-byte unchanged; the
    whole pass is a no-op unless ``DISCOPT_LIFT_LOOSE_PRODUCTS`` is set (the caller
    gates it, but the walker also short-circuits on the flag for safety).

    Sound: ``t == g(x)`` is an exact identity substitution over ``g``'s FBBT box,
    so no feasible point is cut; the rewrite only replaces a *dropped* transcen-
    dental power with a relaxable monomial.
    """
    if not _lift_loose_products_enabled():
        return expr
    lifter.tick()
    # #1565: a subtree with no call power is returned unchanged by this walker
    # (identity-preserving, no lifter call), so skipping it is exact -- and the
    # memoised scan answers that once per shared node instead of once per path.
    if not _scan_for_liftable_call_power(expr, model, lifter.call_power_memo):
        return expr
    if isinstance(expr, BinaryOp):
        call_power = _liftable_call_power_base(expr, model)
        if call_power is not None:
            g_call, n = call_power
            # Recurse into the call argument first (it may hide another such power).
            inner_arg = _prelift_call_powers(g_call.args[0], model, lifter)
            rebuilt_call = (
                g_call if inner_arg is g_call.args[0] else FunctionCall(g_call.func_name, inner_arg)
            )
            t = lifter.expression(rebuilt_call)
            if isinstance(t, Variable):
                return BinaryOp("**", t, Constant(float(n)))
            # Aux could not be bounded (unbounded argument): leave as-is (the term
            # stays dropped exactly as before — never unsound).
            if rebuilt_call is not g_call:
                return BinaryOp("**", rebuilt_call, expr.right)
            return expr
        left = _prelift_call_powers(expr.left, model, lifter)
        right = _prelift_call_powers(expr.right, model, lifter)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    if isinstance(expr, UnaryOp):
        operand = _prelift_call_powers(expr.operand, model, lifter)
        if operand is expr.operand:
            return expr
        return UnaryOp(expr.op, operand)
    if isinstance(expr, FunctionCall):
        new_args = [_prelift_call_powers(a, model, lifter) for a in expr.args]
        if all(na is oa for na, oa in zip(new_args, expr.args)):
            return expr
        return FunctionCall(expr.func_name, *new_args)
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
    lifter.tick()
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
        arg = lifter.call_arg_to_lift(expr)
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
    lifter.tick()
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


def _denominator_sign_slack(
    denom: Expression, model: Model, memo: _BoundMemo | None = None
) -> tuple[float, float]:
    """#1397: how much of each ``_bound_expression(denom)`` endpoint could be round-off.

    Returned as ``(lo_slack, hi_slack)`` in the denominator's own units so each adds
    to :data:`_ZERO_MARGIN` directly. ``inf`` for an endpoint whose interval error
    :func:`bound_expression_error` cannot bound (an unknown call, a non-sign-definite
    nested quotient), which makes that sign test fail -- the clear is refused and the
    McCormick-``lp`` path bounds the division instead. That is the sound direction:
    clearing is an *optional* rewrite, so refusing one costs tightening strength,
    while clearing a sign-indefinite denominator flips the constraint over part of
    the box.

    The endpoints are tracked separately because they fail separately: ``0.01 + x``
    with ``x.ub = +inf`` has an exact lower endpoint and an unusable upper one, and
    its *lower* endpoint is the one the positive-sign test reads.

    ``dmin`` is reduced by the same slack at the call site, because
    :func:`_clear_divisions` divides the cleared body by it to keep the absolute
    feasibility tolerance sound: an over-stated ``dmin`` would under-scale that body
    and let a gross violation slip under the tolerance.
    """
    err_lo, err_hi = bound_expression_error(denom, model, memo)
    return (
        float(err_lo) if np.isfinite(err_lo) else np.inf,
        float(err_hi) if np.isfinite(err_hi) else np.inf,
    )


def _find_clearable_denominator(
    expr: Expression,
    model: Model,
    memo: _BoundMemo | None = None,
    visited: dict[int, Expression] | None = None,
):
    """Return the denominator ``D`` of the first division term ``N/D`` in
    *expr*'s additive structure whose ``D`` is non-constant and sign-definite
    over the variable box, as ``(D, sign, dmin)`` where ``sign`` is +1/-1 and
    ``dmin = min |D|`` over the box.  ``None`` if no such division exists.

    #1565: linear in DAG size. *memo* shares every interval and interval error
    across the denominators examined (and, from :func:`_clear_divisions`, across
    its passes over one box). *visited* records each node already searched: the
    search returns at the first hit, so a node seen before is one whose search
    already came back ``None`` and would again -- skipping it is exact. It maps
    ``id`` to the node itself so an address cannot be recycled mid-search.
    """
    if memo is None:
        memo = _BoundMemo()
    if visited is None:
        visited = {}
    if id(expr) in visited:
        return None
    visited[id(expr)] = expr
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            found = _find_clearable_denominator(expr.left, model, memo, visited)
            if found is not None:
                return found
            return _find_clearable_denominator(expr.right, model, memo, visited)
        if expr.op == "/":
            d = expr.right
            if not isinstance(d, Constant):
                lo, hi = _bound_expression(d, model, memo)
                # #1397: ``_ZERO_MARGIN`` alone assumes ``lo``/``hi`` are exact.
                # They are not -- ``_bound_expression`` is plain float interval
                # arithmetic with no outward rounding -- and clearing a denominator
                # that can in fact change sign FLIPS the inequality over part of
                # the box, which is a false-optimal generator, not a weaker bound.
                # So the yardstick carries that arithmetic's own error; see
                # :func:`bound_expression_error`. Measured (see
                # ``scripts/audit_1397_denominator_sign_margin.py``): ``y + M - M +
                # 1e-8`` over ``y in [-0.5, 1]`` folds to ``lo = 1e-8``, clearing a
                # 1e-9 margin, while its true infimum is ``-0.5``; the gate cleared
                # it at every M from 1e16 to 1e18. An O(1) denominator is
                # unaffected: its error is 0.0 (declared bounds are exact floats).
                lo_slack, hi_slack = _denominator_sign_slack(d, model, memo)
                if lo > _ZERO_MARGIN + lo_slack:
                    return d, 1, lo - lo_slack
                if hi < -_ZERO_MARGIN - hi_slack:
                    return d, -1, -hi - hi_slack
            # Search the numerator for a nested division.
            return _find_clearable_denominator(expr.left, model, memo, visited)
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _find_clearable_denominator(expr.operand, model, memo, visited)
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
    # #1565: one interval memo for every pass. The box is fixed for the whole
    # call (clearing builds expressions; it never touches a variable bound), and
    # the nodes a later pass bounds are the same objects an earlier pass did.
    memo = _BoundMemo()
    for _ in range(8):  # bounded: each pass clears one denominator family
        found = _find_clearable_denominator(body, model, memo)
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

    FAILS CLOSED when the body is too large to distribute within
    :func:`~.term_classifier.distribution_exceeds_budget`.  This guard is the one
    caller of ``distribute_products`` whose "found nothing" answer *enables* a
    rewrite rather than declining one, so a partial distribution cannot be read
    as a clean bill of health.  The exposure is not hypothetical: the degree-2
    test below rejects any term with a sum factor (``_decompose_poly_product``
    files those under ``extra``), and clearing itself wraps the whole body in
    ``scale * body`` whenever ``dmin < 1``, so an undistributed body presents as
    a single product with a sum factor and walks straight past the check —
    re-enabling exactly the gear4-class false infeasibility this exists to stop.
    """
    if distribution_exceeds_budget(body):
        return True
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


def has_factorable_work(model: Model, *, deadline: Optional[Callable[[], bool]] = None) -> bool:
    """True if any constraint/objective has a clearable division or a mixed
    repeated-factor product — i.e. the pass would change the model.

    Exposed so the solver can run this cheap structural scan *before* the more
    expensive convexity classification used to gate the (convexity-destroying)
    rewrite: only nonconvex models that actually have liftable terms pay for
    convexity detection.

    The structural scan recurses one frame per expression node; on a deep
    ``from_nl`` graph that can exceed the default recursion limit, so it runs
    with size-scaled recursion headroom (issue #271).

    ``deadline`` is #1456 item 2: a coarse abstention check, consulted once per
    constraint in the outer loop and never inside the per-expression walk, so
    the check itself is not a cost.  On expiry the scan takes its existing
    "found nothing" path and the caller leaves the model alone.  Without it a
    pass-entry gate is not enough — ``truck`` was measured spending **81.5 s**
    between two consecutive gate checks, i.e. inside this one pass, against
    ``time_limit=10``.
    """
    return _run_factorable_with_headroom(
        model, lambda: _has_factorable_work_inner(model, deadline=deadline)
    )


def _has_factorable_work_inner(
    model: Model, *, deadline: Optional[Callable[[], bool]] = None
) -> bool:
    def scan(expr: Expression) -> bool:
        if _find_clearable_denominator(expr, model) is not None:
            return True
        # TD-A: scan for ``g(x)**n`` on the *pre-distribute* tree — distribution
        # would collapse it into the ``g·g`` product that hides the structure.
        if _lift_loose_products_enabled() and _scan_for_liftable_call_power(expr, model):
            return True
        # #1537 / #1588 review: a translated monomial is factorable work in its
        # own right (pre-distribute, the same walk the prelift makes).
        if _lift_affine_monomials_enabled() and _scan_for_translated_monomial(expr):
            return True
        dist = distribute_products(expr)
        if _scan_for_mixed_product(dist, model):
            return True
        if _scan_for_liftable_call(dist, model):
            return True
        return _scan_for_liftable_fractional_power(dist, model)

    if model._objective is not None and scan(model._objective.expression):
        return True
    for c in model._constraints:
        # #1456 item 2. Abstaining here is the same answer the scan gives when
        # it finds nothing, so it costs structure recognition and never
        # correctness: the caller's response to False is "leave the model
        # alone".
        if deadline is not None and deadline():
            return False
        if isinstance(c, Constraint) and scan(c.body):
            return True
    return False


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

    ``_find_clearable_denominator`` recurses one frame per additive node; on a
    deep ``from_nl`` graph that can exceed the default recursion limit, so the
    scan runs with size-scaled recursion headroom (issue #271).
    """
    return _run_factorable_with_headroom(
        model,
        lambda: any(
            isinstance(c, Constraint) and _find_clearable_denominator(c.body, model) is not None
            for c in model._constraints
        ),
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


def _scan_for_liftable_call_power(
    expr: Expression, model: Model, memo: dict[int, tuple[Expression, bool]] | None = None
) -> bool:
    """True if *expr* contains an integer power ``g(x)**n`` (n >= 2) of a univariate
    transcendental call — the TD-A lift target. Scans the *pre-distribute* tree
    (``distribute_products`` would collapse ``g(x)**2`` into ``g·g`` and hide it).

    #1565: memoised by node identity (the node is held, so its ``id`` cannot be
    recycled), so a shared subexpression is scanned once. The verdict is a pure
    function of the node, so a memo hit is exactly the recomputed answer. It
    descends the same edges ``_prelift_call_powers`` does, which is what lets that
    walker skip a subtree this scan clears.
    """
    if memo is None:
        memo = {}
    hit = memo.get(id(expr))
    if hit is not None:
        return hit[1]
    if _liftable_call_power_base(expr, model) is not None:
        found = True
    elif isinstance(expr, BinaryOp):
        found = _scan_for_liftable_call_power(
            expr.left, model, memo
        ) or _scan_for_liftable_call_power(expr.right, model, memo)
    elif isinstance(expr, UnaryOp):
        found = _scan_for_liftable_call_power(expr.operand, model, memo)
    elif isinstance(expr, FunctionCall):
        found = any(_scan_for_liftable_call_power(a, model, memo) for a in expr.args)
    else:
        found = False
    memo[id(expr)] = (expr, found)
    return found


def _scan_for_liftable_fractional_power(expr: Expression, model: Model) -> bool:
    """True if *expr* contains a fractional power ``base**p`` (non-integer *p*)
    over a *composite* base — a product, sum/affine, or other multi-leaf
    sub-expression — but NOT a single variable or an integer power of one.

    ``_lift_objective_atoms`` lifts such a node to ``d == t**p`` with ``t == base``
    an auxiliary variable (a bilinear/monomial/affine aux for the base, then a
    fractional-power aux for the outer power), which the McCormick pipeline relaxes
    via the same single-variable fractional-power envelope that ``sqrt(t)`` gets.
    But that lift is only *applied* when ``has_factorable_work`` triggers the
    reform pass, and the existing scanners look only for mixed repeated-factor
    products and sqrt/exp *calls*. A fractional power written in **power form** —
    ``(x*y)**0.5`` (product base) or ``(x+y)**0.5`` (affine base) — is none of
    those, so the gate returned False, the pass never ran, and the term either
    dropped from the relaxation (product base → frozen bound, the ex1226 failure
    mode) or got a loose envelope (affine base → never closed: ``(x+y)**0.5``
    churned 3503 nodes while the equivalent *call* ``sqrt(x+y)`` closed in 251).

    Only the **power form** is matched here: ``sqrt(...)``/``exp(...)`` *calls* are
    FunctionCall nodes reached by the composite-univariate envelope (and
    ``_scan_for_liftable_call``), so they are untouched. A single-variable base
    (``x**0.5``) is relaxed natively via ``fractional_power_var_map`` and is
    excluded by ``_is_simple_power_base``.
    """
    if isinstance(expr, BinaryOp):
        if (
            expr.op == "**"
            and isinstance(expr.right, Constant)
            and float(expr.right.value) != int(float(expr.right.value))
            and not _is_simple_power_base(expr.left, model)
            and len(_collect_variables(expr.left)) >= 1
        ):
            return True
        return _scan_for_liftable_fractional_power(
            expr.left, model
        ) or _scan_for_liftable_fractional_power(expr.right, model)
    if isinstance(expr, UnaryOp):
        return _scan_for_liftable_fractional_power(expr.operand, model)
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
    behavioural change). An exception is a defect and propagates (#1520).

    The rewrite recurses one frame per expression node, so it runs with the same
    size-scaled recursion headroom as :func:`factorable_reformulate` (#1520: a
    3000-term ``.nl`` row used to ``RecursionError``, which a blanket handler
    turned into "no entropy term found").
    """
    return _run_factorable_with_headroom(model, lambda: _canonicalize_entropy_inner(model))


def _canonicalize_entropy_inner(model: Model) -> Model:
    # #1520: no except. The walk declines by returning the model unchanged; the
    # old ``except Exception: return model`` could only hide a defect as "no
    # entropy term found". ``ComplementarityProvenanceError`` already propagated.
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
    new_model._rebuild_name_index()  # keep the name cache in sync (M7)
    new_model._objective = new_objective
    new_model._constraints = new_constraints
    # Complementarity provenance (#1147): forward the relation set onto the
    # rebuilt model. An unresolvable relation raises rather than degrading to a
    # silent drop.
    carry_complementarities(model, new_model, pass_name="entropy canonicalization")
    carry_validation_guards(model, new_model)  # #1498
    return new_model


def factorable_reformulate(
    model: Model,
    *,
    clear_only: bool = False,
    deadline: Optional[Callable[[], bool]] = None,
) -> Model:
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

    The rewrite walkers recurse one frame per expression node; on a deep
    ``from_nl`` graph that can exceed the default recursion limit, so the whole
    pass runs with size-scaled recursion headroom (issue #271).

    ``deadline`` (#1456 item 2) is checked once per constraint in the rebuild
    loop.  On expiry the pass abandons **wholesale** and returns the original
    model: the half-built replacement is discarded rather than returned, so an
    abstention is always the documented "returned unchanged" outcome and never
    a partially rewritten model.  That is what makes a clock admissible here —
    it selects between two states the pass already produces, and cannot invent
    a third.  It is not even a new escape: the rebuild loop has always been able
    to bail to the original model from any point inside itself (the explicit
    ``return model`` declines), so abandoning is an existing, exercised return
    path reached for a new reason.
    """
    return _run_factorable_with_headroom(
        model,
        lambda: _factorable_reformulate_inner(model, clear_only=clear_only, deadline=deadline),
    )


def _factorable_reformulate_inner(
    model: Model,
    *,
    clear_only: bool = False,
    deadline: Optional[Callable[[], bool]] = None,
) -> Model:
    # #1520: no catch-all. Every decline is an explicit ``return model`` (no work,
    # the deadline, the soundness gates); the old ``except Exception: return model``
    # could only hide a defect in the rewrite as "nothing to reformulate". The one
    # ``except`` below catches only the private ``_LiftAbandoned`` sentinel.
    if not _has_factorable_work_inner(model, deadline=deadline):
        return model

    new_model = Model(model.name)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._rebuild_name_index()  # keep the name cache in sync (M7)
    new_model._objective = model._objective

    lifter = _Lifter(new_model, deadline=deadline)
    # #1565: the deadline is also consulted inside a constraint's lift (see
    # ``_Lifter.tick``); on expiry everything built so far is dropped, exactly as
    # the per-constraint check below drops it.
    try:
        rebuilt = _rebuild_and_lift(model, new_model, lifter, clear_only, deadline)
    except _LiftAbandoned:
        return model
    if rebuilt is None:
        return model

    # Defining equalities for the aux variables come first so downstream
    # bound propagation sees them early.
    new_model._constraints = lifter.aux_constraints + rebuilt
    # R4: surface the zero-spanning product-factor auxes (if any were tagged
    # under the flag) so the solver can keep them branchable. Always set the
    # attribute (empty by default) for a stable, easy-to-read contract.
    new_model._zero_spanning_factor_auxes = set(lifter.zero_spanning_factor_auxes)
    new_model._implied_integer_auxes = set(getattr(model, "_implied_integer_auxes", ())) | set(
        lifter.implied_integer_auxes
    )
    # Complementarity provenance (#1147). The lifts rewrite constraint
    # *bodies*; the relation's source operands are untouched and still read
    # the shared Variable objects, so the relation set forwards intact.
    carry_complementarities(model, new_model, pass_name="factorable reformulation")
    carry_validation_guards(model, new_model)  # #1498
    return new_model


def _rebuild_and_lift(
    model: Model,
    new_model: Model,
    lifter: _Lifter,
    clear_only: bool,
    deadline: Optional[Callable[[], bool]],
) -> list[Constraint] | None:
    """The rebuild loop of :func:`_factorable_reformulate_inner`: returns the
    rebuilt constraints (and lifts the objective of *new_model* in place), or
    ``None`` when the per-constraint deadline fired. ``_LiftAbandoned`` from
    inside a lift propagates to the caller."""
    rebuilt: list[Constraint] = []
    for c in model._constraints:
        # #1456 item 2. Wholesale abandonment: ``new_model`` and everything
        # ``lifter`` has built for it are dropped on the floor, and the
        # caller gets the model it passed in. A partial rewrite would be a
        # third state nothing downstream is written against.
        if deadline is not None and deadline():
            return None
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
        body = _prelift_call_powers(body, new_model, lifter)
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
        obj_expr = _prelift_call_powers(new_model._objective.expression, new_model, lifter)
        obj_expr = _prelift_blowup_products(obj_expr, new_model, lifter)
        obj_expr = distribute_products(obj_expr)
        obj_expr = _lift_objective_atoms(obj_expr, new_model, lifter)
        lifted_obj = _lift_expr(obj_expr, new_model, lifter)
        if lifted_obj is not new_model._objective.expression:
            from discopt.modeling.core import Objective

            new_model._objective = Objective(lifted_obj, new_model._objective.sense)

    return rebuilt
