"""Integer-factor bilinear reformulation: relax ``x_i * x_j`` *exactly* when a
factor is an integer variable.

A single continuous McCormick envelope of a bilinear term ``x_i * x_j`` over a
wide box is loose — its integer optimum can sit strictly below the true optimum
(e.g. the ``ex126x`` trim-loss family: discopt's relaxation caps at 19.1 while
the true optimum is 19.6). When one factor is an *integer* variable
``x_i in [lo, hi]`` it can be binary-expanded,

    x_i = lo + sum_k 2^k e_k        (e_k binary,  k = 0 .. ceil(log2(hi-lo+1))-1)

so the product becomes

    x_i * x_j = lo * x_j + sum_k 2^k (e_k * x_j),

and each ``binary x variable`` product ``e_k * x_j`` is lifted to an auxiliary
``v_k`` with its **exact big-M linearization** (``v_k <= U*e_k``,
``v_k <= x_j``, ``v_k >= x_j - U*(1-e_k)``, ``v_k >= 0``). The result is a purely
**linear** model — every bilinear term is gone, so the bilinear MINLP becomes an
equivalent pure MILP that discopt's MILP branch-and-bound solves directly. The
big-M is exact at ``e_k in {0,1}`` (no McCormick gap), so the MILP optimum equals
the true MINLP optimum.

The rewrite is value-preserving: ``x_i = lo + sum 2^k e_k`` reproduces every
integer value of ``x_i`` over ``[lo, hi]`` (combinations exceeding ``hi`` are
ruled out by ``x_i``'s own upper bound), and the product expansion is an
algebraic identity. Only the *relaxation* changes (loose -> exact). The pass is a
no-op when no integer-factor bilinear term exists, so it never regresses a model.

Note: this triggers on *declared* integer factors. Models whose factors are
implied-integer (declared continuous but forced integer by the constraints, as
in ``ex1263``) need the implied-integer detection pass to mark them integer
first; this module is that detector's downstream consumer.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    IndexExpression,
    Model,
    Variable,
    VarType,
)
from discopt.solver_tuning import _env_flag

from .factorable_reform import _collect_mul_factors
from .term_classifier import _compute_var_offset, distribute_products

# Skip an integer factor whose range needs more than this many bits — the
# expansion adds one binary and one aux product per bit, so a huge range would
# blow up the model for no practical gain (and such a variable is better left to
# spatial branching). 12 bits covers ranges up to 4095.
_MAX_BITS = 12
# Beyond this magnitude a big-M coefficient is numerically meaningless (matches the
# solver's large-bound warning threshold); products with such a factor are not
# big-M-linearizable and the reformulation aborts (issue #286).
_BIGM_BOUND_CAP = 1e15


def _scalar_var_ref(expr: Expression) -> Optional[tuple[Variable, int]]:
    """Return ``(variable, flat_element)`` if *expr* is a scalar variable
    reference (a scalar ``Variable`` or an ``IndexExpression`` selecting one
    scalar element), else ``None``."""
    if isinstance(expr, Variable):
        if expr.size == 1:
            return expr, 0
        return None
    if isinstance(expr, IndexExpression):
        var = expr.variable if hasattr(expr, "variable") else getattr(expr, "var", None)
        if not isinstance(var, Variable):
            return None
        idx = getattr(expr, "index", None)
        if isinstance(idx, int):
            return var, idx
        # Multi-dim or non-scalar index: not handled here.
        return None
    return None


def _int_factor_range(
    var: Variable, elem: int, implied: "frozenset[tuple[int, int]]" = frozenset()
) -> Optional[tuple[int, int]]:
    """Return integer ``(lo, hi)`` if ``var[elem]`` is integer-valued (declared
    ``INTEGER``, or *implied-integer* per the ``implied`` set) with finite bounds
    spanning a range expressible in ``_MAX_BITS`` bits, else ``None``. Binary
    variables are already exact under McCormick, so they are excluded."""
    if var.var_type != VarType.INTEGER and (var._index, elem) not in implied:
        return None
    lo = float(np.asarray(var.lb).flat[elem])
    hi = float(np.asarray(var.ub).flat[elem])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
        return None
    lo_i, hi_i = int(math.floor(lo)), int(math.ceil(hi))
    span = hi_i - lo_i
    if span <= 0 or span > (1 << _MAX_BITS) - 1:
        return None
    return lo_i, hi_i


class _Expander:
    """Creates and caches the binary expansion of integer factor variables and
    accumulates the linking ``x_i == lo + sum 2^k e_k`` constraints."""

    def __init__(self, model: Model, implied=frozenset(), participation=None, multilinear=False):
        self.model = model
        self.implied = implied
        # When True, also exact-linearize integer-multilinear products (>=3 factors,
        # <=1 continuous) — issue #707. Left off for the pure-bilinear entry points.
        self.multilinear = multilinear
        # (var._index, elem) -> number of bilinear products the factor appears in;
        # the more-shared factor is expanded so its bits are reused (fewer vars).
        self.participation = participation or {}
        self.aux_constraints: list[Constraint] = []
        # (var._index, elem) -> (lo, [(coef, e_k Variable), ...])
        self._cache: dict[tuple[int, int], tuple[int, list[tuple[int, Variable]]]] = {}
        self._counter = 0
        # Warm-start reconstruction spec: one entry per auxiliary column, in the
        # exact order the columns are appended to ``model._variables`` (so the aux
        # block is a suffix of the reformed flat layout and each entry's index ==
        # its append position). Every aux is *exactly determined* by the original
        # variables, so a warm start over the originals extends to the reformed
        # vector by evaluating this spec in order (see ``extend_initial_point``).
        #   ("bit", (var._index, elem), k, lo, nbits): bit k of round(x - lo).
        #   ("prod", (e._index, 0), (other._index, other_elem)): v = e * other.
        self.aux_spec: list[tuple] = []
        # Cleared if an aux column cannot be described (defensive): then the whole
        # spec is discarded and no warm start is mapped rather than one misaligned.
        self.spec_ok = True
        # Caches for the multilinear path (issue #707) so a binary monomial shared
        # across several product terms is lifted once: key -> aux Variable.
        #   _and_cache: tuple(sorted bit _index) -> binary AND aux
        #   _bigm_cache: (e._index, other._index, other_elem) -> big-M product aux
        self._and_cache: dict[tuple, Variable] = {}
        self._bigm_cache: dict[tuple, Variable] = {}
        # Configuration metadata for the disjunctive config bound (#732 Stage 2).
        # Collected on the multilinear path only: the flat indices of the
        # integer factors of every exact-linearized product, split into
        # *indicator-like* factors (range {0,1} — the configuration selectors)
        # and *count-like* factors (range span >= 2 — e.g. pump counts). Pure
        # metadata: never enters a bound or feasibility test here; the
        # disjunctive pass uses it to enumerate/peel configuration boxes.
        self.config_indicator_flats: set[int] = set()
        self.config_count_flats: set[int] = set()
        # Coupling-RLT (issue #721, default-OFF ``DISCOPT_MULTILINEAR_COUPLING_RLT``).
        # For a continuous-times-AND product ``v = z*c`` with ``z = AND(bits)`` and a
        # non-negative continuous factor ``c``, the plain big-M envelope of ``v``
        # decouples in the LP: with the bits/``z`` fractional the objective's ``c*z``
        # cost can relax to 0 regardless of ``c`` (ex1252 loosest node: the reformed
        # ``x15*(x0*x3*x18)`` term relaxes to 0 though the cubic rows force
        # ``x15 >= 12.44``, pinning the bound at the objective constant 12658). The
        # RLT rows ``v <= b*c`` and ``v >= sum_b (b*c) - (n-1)*c`` (products of the
        # exact AND hull ``z <= b`` / ``z >= sum b - (n-1)`` with ``c >= 0``) tie ``v``
        # back to ``c``, lifting the bound sound-ly (entry experiment: 12658 ->
        # ~57435 at that node). See docs/dev/performance-plan.md §6.
        self.coupling_rlt = _env_flag("DISCOPT_MULTILINEAR_COUPLING_RLT", default=False)

    def expansion(self, var: Variable, elem: int, lo: int, hi: int):
        key = (var._index, elem)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        nbits = max(1, math.ceil(math.log2(hi - lo + 1)))
        bits: list[tuple[int, Variable]] = []
        ref = var if var.size == 1 else IndexExpression(var, elem)
        # x_i - lo - sum 2^k e_k == 0
        link = ref if lo == 0 else BinaryOp("-", ref, Constant(float(lo)))
        for k in range(nbits):
            e = Variable(f"_ipx_e{self._counter}", VarType.BINARY, (), 0.0, 1.0, self.model)
            self._counter += 1
            self.model._variables.append(e)
            # e_k is bit k of round(x_i - lo): the link row pins sum 2^k e_k to
            # x_i - lo exactly, and the binary digits of a value in [0, 2^nbits-1]
            # are unique.
            self.aux_spec.append(("bit", (var._index, elem), k, lo, nbits))
            bits.append((1 << k, e))
            term = e if k == 0 else BinaryOp("*", Constant(float(1 << k)), e)
            link = BinaryOp("-", link, term)
        self.aux_constraints.append(Constraint(link, "==", 0.0))
        self._cache[key] = (lo, bits)
        return lo, bits

    def bigm_product(self, e: Variable, other: Expression, lo_o: float, hi_o: float) -> Variable:
        """Return an aux ``v == e * other`` (``e`` binary, ``other in [lo_o, hi_o]``)
        via the **exact** big-M linearization

            v <= hi_o * e,   v >= lo_o * e,
            v <= other - lo_o*(1-e),   v >= other - hi_o*(1-e).

        At ``e in {0,1}`` these force ``v = 0`` (e=0) or ``v = other`` (e=1), so the
        product is reproduced exactly — no McCormick gap, and no bilinear term
        remains in the model."""
        # The big-M coefficients ARE the other factor's bounds; an infinite (or
        # astronomically large) bound makes the rows ``v <= hi_o*e`` /
        # ``v >= other - hi_o*(1-e)`` vacuous, so the aux ``v`` (hence the product) is
        # left unbounded and the reformulated "MILP" is spuriously unbounded though
        # the original problem is bounded (carton7, issue #286). Abort the whole
        # pass; the caller's guard returns the original model and the solver keeps
        # the (bound-aware) spatial path. ``_BIGM_BOUND_CAP`` matches discopt's
        # large-bound warning threshold — beyond it the big-M is numerically
        # meaningless, not just at true infinity.
        if not (abs(lo_o) <= _BIGM_BOUND_CAP and abs(hi_o) <= _BIGM_BOUND_CAP):
            raise ValueError("cannot big-M-linearize a product with an unbounded factor")
        # Reuse an identical ``e * other`` product if one was already lifted (the
        # multilinear expansion can request the same AND-aux times continuous
        # factor across several monomials). Keyed on the scalar refs. Scoped to the
        # multilinear path so the pure-bilinear reform stays byte-identical (its
        # blowup-guard threshold depends on the exact aux count — issue #707).
        _oref = _scalar_var_ref(other) if self.multilinear else None
        _ckey = (e._index, _oref[0]._index, _oref[1]) if _oref is not None else None
        if _ckey is not None and _ckey in self._bigm_cache:
            return self._bigm_cache[_ckey]
        v = Variable(
            f"_ipx_v{self._counter}",
            VarType.CONTINUOUS,
            (),
            min(0.0, lo_o),
            max(0.0, hi_o),
            self.model,
        )
        self._counter += 1
        self.model._variables.append(v)
        # v == e * other, both scalar variable references (``other`` is either an
        # original factor or, in the square expansion, another expansion bit — in
        # every case already appended before this product), so v reconstructs as
        # value(e) * value(other) at any point. Record it for warm-start mapping.
        oref = _scalar_var_ref(other)
        if oref is None:  # pragma: no cover - defensive; other is always a scalar ref
            self.spec_ok = False
        else:
            self.aux_spec.append(("prod", (e._index, 0), (oref[0]._index, oref[1])))
        ac = self.aux_constraints
        # discopt's LP extractor folds the RHS into the body (``body sense 0``);
        # every row below is therefore normalized to rhs == 0 with the constant
        # carried inside the body (a nonzero ``Constraint`` rhs is silently
        # dropped by ``extract_lp_data`` — see _extract_constraints_algebraic).
        # v - hi_o*e <= 0
        ac.append(Constraint(BinaryOp("-", v, BinaryOp("*", Constant(hi_o), e)), "<=", 0.0))
        # v - lo_o*e >= 0
        ac.append(Constraint(BinaryOp("-", v, BinaryOp("*", Constant(lo_o), e)), ">=", 0.0))
        # v - other - lo_o*e + lo_o <= 0   (i.e. v <= other - lo_o*(1-e))
        body_u = BinaryOp("-", BinaryOp("-", v, other), BinaryOp("*", Constant(lo_o), e))
        ac.append(Constraint(BinaryOp("+", body_u, Constant(lo_o)), "<=", 0.0))
        # v - other - hi_o*e + hi_o >= 0   (i.e. v >= other - hi_o*(1-e))
        body_l = BinaryOp("-", BinaryOp("-", v, other), BinaryOp("*", Constant(hi_o), e))
        ac.append(Constraint(BinaryOp("+", body_l, Constant(hi_o)), ">=", 0.0))
        if _ckey is not None:
            self._bigm_cache[_ckey] = v
        return v

    def and_product(self, bits: "list[Variable]") -> Variable:
        """Return a **binary** aux ``z = prod_i e_i`` for a set of binary variables
        ``e_i`` via the exact multilinear-AND hull

            z <= e_i  (each i),   z >= sum_i e_i - (n-1),   z in {0,1}.

        At ``e_i in {0,1}`` these force ``z = 1`` iff every ``e_i = 1`` — the exact
        (integral) convex hull of the binary product, tighter than a term-wise
        McCormick envelope. Shared monomials are lifted once (``_and_cache``)."""
        key = tuple(sorted(b._index for b in bits))
        cached = self._and_cache.get(key)
        if cached is not None:
            return cached
        n = len(bits)
        z = Variable(f"_ipx_z{self._counter}", VarType.BINARY, (), 0.0, 1.0, self.model)
        self._counter += 1
        self.model._variables.append(z)
        # z is exactly the product of its binary inputs, so it reconstructs from a
        # point over the originals once its inputs are reconstructed. Record the
        # AND spec (evaluated after every input bit, which are appended earlier).
        self.aux_spec.append(("and", (z._index, 0), [(b._index, 0) for b in bits]))
        ac = self.aux_constraints
        # z - e_i <= 0  (each factor upper-bounds the product)
        for b in bits:
            ac.append(Constraint(BinaryOp("-", z, b), "<=", 0.0))
        # z - sum_i e_i + (n-1) >= 0  (bits is non-empty: n >= 2 at every call site)
        s: Expression = bits[0]
        for b in bits[1:]:
            s = BinaryOp("+", s, b)
        body = BinaryOp("+", BinaryOp("-", z, s), Constant(float(n - 1)))
        ac.append(Constraint(body, ">=", 0.0))
        self._and_cache[key] = z
        return z

    def add_coupling_rlt(
        self, v: Variable, bits: "list[Variable]", other: Expression, lo_o: float, hi_o: float
    ) -> None:
        """Add RLT rows tying ``v = AND(bits) * other`` to the per-bit products.

        Let ``z = AND(bits)`` (so ``v == z*other``). The exact AND hull gives
        ``z <= b`` for each bit ``b`` and ``z >= sum_b b - (n-1)``. Multiplying each
        by a **non-negative** ``other`` (``lo_o >= 0``) yields the valid RLT rows

            v <= b*other            (each bit b)
            v >= sum_b (b*other) - (n-1)*other

        where each ``b*other`` is the exact big-M product (reused from the cache when
        already lifted). These hold at every integral point, so they never cut a
        feasible solution, and they tighten the LP envelope of ``v`` toward ``z*other``
        — closing the objective-coupling leak (issue #721). No-op unless the flag is
        on, ``other`` is sign-definite non-negative, and the AND is genuinely
        multi-bit (a single-bit ``z`` already *is* its big-M product, exactly)."""
        if not self.coupling_rlt or lo_o < 0.0 or len(bits) < 2:
            return
        ws = [self.bigm_product(b, other, lo_o, hi_o) for b in bits]
        ac = self.aux_constraints
        # v - b*other <= 0   (v <= b*other), for each bit
        for w in ws:
            ac.append(Constraint(BinaryOp("-", v, w), "<=", 0.0))
        # v - sum_b (b*other) + (n-1)*other >= 0   (v >= sum_b (b*other) - (n-1)*other)
        s: Expression = ws[0]
        for w in ws[1:]:
            s = BinaryOp("+", s, w)
        body = BinaryOp(
            "+", BinaryOp("-", v, s), BinaryOp("*", Constant(float(len(bits) - 1)), other)
        )
        ac.append(Constraint(body, ">=", 0.0))

    def add_bitlink_rlt(
        self,
        ref: Expression,
        lo: int,
        bits: "list[tuple[int, Variable]]",
        other: Expression,
        lo_o: float,
        hi_o: float,
    ) -> None:
        """RLT of the bit-linking equality ``x_i = lo + sum_k 2^k e_k`` times ``other``.

        Multiplying that exact identity by the continuous factor ``other`` gives the
        valid (exact) linear relation

            x_i*other == lo*other + sum_k 2^k (e_k*other)

        where each ``e_k*other`` is the exact big-M product and ``x_i*other`` is left
        as a bilinear term the McCormick relaxer envelopes. This is what actually
        closes the coupling leak: at any node where ``x_i`` is fixed, ``x_i*other``
        is McCormick-exact, so the per-bit products ``e_k*other`` are pinned (the
        bit-linking row alone leaves the ``e_k`` fractional in the LP, which is why
        the AND-hull RLT above is not enough on its own). Requires ``other >= 0``
        (``lo_o >= 0``) so the objective coupling — the only current caller — is
        sign-definite; a sign-mixed factor is skipped (left to plain McCormick)."""
        if not self.coupling_rlt or lo_o < 0.0:
            return
        acc: Expression = BinaryOp("*", ref, other)
        if lo != 0:
            acc = BinaryOp("-", acc, BinaryOp("*", Constant(float(lo)), other))
        for ck, ek in bits:
            w = self.bigm_product(ek, other, lo_o, hi_o)
            term = w if ck == 1 else BinaryOp("*", Constant(float(ck)), w)
            acc = BinaryOp("-", acc, term)
        self.aux_constraints.append(Constraint(acc, "==", 0.0))


def _expand_product(
    lo: int, bits, other: Expression, lo_o: float, hi_o: float, exp: "_Expander"
) -> Expression:
    """Build ``lo*other + sum_k 2^k v_k`` with ``v_k == e_k*other`` big-M-lifted —
    a purely *linear* expression (no bilinear term survives)."""
    out: Optional[Expression] = None
    if lo != 0:
        out = BinaryOp("*", Constant(float(lo)), other)
    for coef, e in bits:
        v = exp.bigm_product(e, other, lo_o, hi_o)
        term = v if coef == 1 else BinaryOp("*", Constant(float(coef)), v)
        out = term if out is None else BinaryOp("+", out, term)
    return out if out is not None else Constant(0.0)


def _expand_square(lo: int, bits, exp: "_Expander") -> Expression:
    """Build the exact linear form of ``x^2`` for ``x = lo + sum_k c_k e_k`` (``e_k``
    binary, ``c_k = 2^k``):

        x^2 = lo^2 + sum_k (2*lo*c_k + c_k^2) e_k + 2 sum_{k<j} c_k c_j (e_k e_j)

    using ``e_k^2 = e_k`` (binary) and lifting each binary-AND ``e_k e_j`` via the
    exact big-M product. Purely linear — no monomial term survives."""
    out: Optional[Expression] = None
    if lo != 0:
        out = Constant(float(lo * lo))
    for ck, ek in bits:
        coef = 2.0 * lo * ck + ck * ck
        term = BinaryOp("*", Constant(coef), ek)
        out = term if out is None else BinaryOp("+", out, term)
    for a in range(len(bits)):
        ca, ea = bits[a]
        for b in range(a + 1, len(bits)):
            cb, eb = bits[b]
            w = exp.bigm_product(ea, eb, 0.0, 1.0)  # binary AND e_a*e_b, exact
            term = BinaryOp("*", Constant(2.0 * ca * cb), w)
            out = term if out is None else BinaryOp("+", out, term)
    return out if out is not None else Constant(0.0)


def _try_expand_square(node: BinaryOp, exp: "_Expander") -> Optional[Expression]:
    """If *node* is ``x**2`` with ``x`` an integer scalar variable, return its exact
    binary-expansion linearization, else ``None``. (Higher powers are left to the
    monomial relaxation; only the square is handled exactly here.)"""
    if not (isinstance(node.right, Constant) and abs(float(node.right.value) - 2.0) < 1e-12):
        return None
    ref = _scalar_var_ref(node.left)
    if ref is None:
        return None
    rng = _int_factor_range(ref[0], ref[1], exp.implied)
    if rng is None:
        return None
    lo, bits = exp.expansion(ref[0], ref[1], rng[0], rng[1])
    return _expand_square(lo, bits, exp)


def _try_expand_mul(node: BinaryOp, model: Model, exp: _Expander) -> Optional[Expression]:
    """If *node* is a product whose only variable factors are two distinct scalar
    variable references, one of them an integer, return the exact expansion
    (carrying any constant coefficient factors), else ``None``."""
    factors = _collect_mul_factors(node)
    const = 1.0
    var_refs: list[tuple[Expression, Variable, int]] = []
    for f in factors:
        if isinstance(f, Constant):
            const *= float(f.value)
            continue
        ref = _scalar_var_ref(f)
        if ref is None:
            return None  # a non-scalar / non-constant factor — leave to McCormick
        var_refs.append((f, ref[0], ref[1]))
    if len(var_refs) != 2:
        return None
    (e0, v0, el0), (e1, v1, el1) = var_refs
    if v0._index == v1._index and el0 == el1:
        # x*x square: exact-linearize when x is an integer scalar variable.
        rng = _int_factor_range(v0, el0, exp.implied)
        if rng is None:
            return None
        lo, bits = exp.expansion(v0, el0, rng[0], rng[1])
        sq = _expand_square(lo, bits, exp)
        return BinaryOp("*", Constant(const), sq) if const != 1.0 else sq
    # Each expanded bit adds one big-M aux per product, so expanding the
    # smaller-range factor (fewer bits) minimizes added variables; use product
    # sharing only as a tiebreaker (a cached factor avoids re-adding its bits).
    cands = []
    for (ei, vi, eli), (eo, vo, elo) in (
        ((e0, v0, el0), (e1, v1, el1)),
        ((e1, v1, el1), (e0, v0, el0)),
    ):
        rng = _int_factor_range(vi, eli, exp.implied)
        if rng is not None:
            share = exp.participation.get((vi._index, eli), 0)
            cands.append((rng[1] - rng[0], -share, (vi, eli, rng), (eo, vo, elo)))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1]))
    _, _, (vi, eli, (lo, hi)), (other_e, vo, elo) = cands[0]
    base_lo, bits = exp.expansion(vi, eli, lo, hi)
    lo_o = float(np.asarray(vo.lb).flat[elo])
    hi_o = float(np.asarray(vo.ub).flat[elo])
    expanded = _expand_product(base_lo, bits, other_e, lo_o, hi_o, exp)
    # #732 Stage 2: the integer-BILINEAR expansion has the same L2 decoupling as
    # the multilinear one — with the bits fractional in the LP, every
    # ``v_k = e_k·other`` big-M product can relax to 0 (ex1252 multi-line
    # configs: the pump-count couplings ``x9·x3 = 400·x18`` all collapse,
    # pinning those config bounds at ~0). Tie the bit-linking equality to the
    # other factor exactly as ``_try_expand_multilinear`` does. Multilinear-path
    # only (its ``_bigm_cache`` reuses the products just created — no new
    # columns; the pure-bilinear entry stays byte-identical), same default-OFF
    # flag (``DISCOPT_MULTILINEAR_COUPLING_RLT``).
    if exp.multilinear and exp.coupling_rlt:
        ref_i = vi if vi.size == 1 else IndexExpression(vi, eli)
        exp.add_bitlink_rlt(ref_i, base_lo, bits, other_e, lo_o, hi_o)
    if const != 1.0:
        expanded = BinaryOp("*", Constant(const), expanded)
    return expanded


def _classify_multilinear_factors(node: BinaryOp, exp: _Expander):
    """Split a product *node* into ``(const, int_refs, cont_factors)`` where
    ``int_refs`` is a list of ``(var, elem, lo, hi)`` integer variable factors (with
    multiplicity, one per occurrence) and ``cont_factors`` is a list of
    ``(scalar_expr, lo, hi)`` for the continuous variable factors. **No expansion
    columns are created** — this is a pure inspection so a term that fails the
    reformability guard leaves the model untouched. Returns ``None`` when a factor
    is not a scalar variable / constant, or there are fewer than three variable
    factors (the bilinear/square cases are handled elsewhere)."""
    factors = _collect_mul_factors(node)
    const = 1.0
    var_refs: list[tuple[Expression, Variable, int]] = []
    for f in factors:
        if isinstance(f, Constant):
            const *= float(f.value)
            continue
        ref = _scalar_var_ref(f)
        if ref is None:
            return None  # a non-scalar / non-constant factor — leave to McCormick
        var_refs.append((f, ref[0], ref[1]))
    if len(var_refs) < 3:
        return None
    int_refs: list[tuple[Variable, int, int, int]] = []
    cont_factors: list[tuple[Expression, float, float]] = []
    for e, v, el in var_refs:
        rng = _int_factor_range(v, el, exp.implied)
        if rng is not None:
            int_refs.append((v, el, rng[0], rng[1]))
        else:
            lo = float(np.asarray(v.lb).flat[el])
            hi = float(np.asarray(v.ub).flat[el])
            cont_factors.append((e, lo, hi))
    return const, int_refs, cont_factors


def _try_expand_multilinear(node: BinaryOp, exp: _Expander) -> Optional[Expression]:
    """Exactly linearize an *integer-multilinear* product (>=3 variable factors,
    every factor but at most one integer/binary-valued). Binary-expand each integer
    factor, distribute the product into binary monomials (using ``e^2 = e``), lift
    each monomial to its exact hull (an ``and_product`` aux, or the bit itself for a
    single-bit monomial), and — for the lone continuous factor, if any —
    big-M-lift ``continuous * monomial``. Returns a purely linear expression, or
    ``None`` when the term is not of this class (e.g. two continuous factors → a
    genuine continuous nonlinearity left to McCormick)."""
    split = _classify_multilinear_factors(node, exp)
    if split is None:
        return None
    const, int_refs, cont_factors = split
    # At most one continuous factor (to the first power): two distinct or a repeated
    # continuous factor is a genuine continuous bilinear/square — not exact-linear.
    # Guard BEFORE creating any expansion column so a bailed term is a true no-op.
    if len(cont_factors) > 1 or not int_refs:
        return None
    int_factors = [exp.expansion(v, el, lo, hi) for (v, el, lo, hi) in int_refs]

    # #732 Stage 2: record the configuration structure of this product (pure
    # metadata for the disjunctive config bound — never enters a bound here).
    for v, el, lo_i, hi_i in int_refs:
        flat = _compute_var_offset(v, exp.model) + el
        if lo_i == 0 and hi_i == 1:
            exp.config_indicator_flats.add(flat)
        elif hi_i - lo_i >= 2:
            exp.config_count_flats.add(flat)

    # Issue #721 (default-OFF coupling RLT): tie each integer factor's bit-linking
    # equality to the continuous factor, so the per-bit products ``e_k*c`` are pinned
    # once ``x_i`` is fixed (the level that actually closes the objective-coupling
    # leak). Done once per (integer factor, continuous factor) pair, before the
    # per-monomial AND-hull RLT below.
    if exp.coupling_rlt and cont_factors:
        _cexpr, _clo, _chi = cont_factors[0]
        for (v, el, _lo0, _hi0), (lo_i, bits_i) in zip(int_refs, int_factors):
            ref_i = v if v.size == 1 else IndexExpression(v, el)
            exp.add_bitlink_rlt(ref_i, lo_i, bits_i, _cexpr, _clo, _chi)

    # Distribute the product of the integer factors into monomials over the binary
    # bits. ``poly`` maps a tuple of distinct bit Variables (sorted by index) to its
    # integer coefficient; ``e^2 = e`` collapses repeated bits (repeated integer
    # factor) via the ``any(b is ek ...)`` idempotence check.
    poly: dict[tuple, float] = {(): 1.0}
    for lo, bits in int_factors:
        newpoly: dict[tuple, float] = {}
        for key, coef in poly.items():
            if lo != 0.0:
                newpoly[key] = newpoly.get(key, 0.0) + coef * float(lo)
            for ck, ek in bits:
                nk = (
                    key
                    if any(b is ek for b in key)
                    else tuple(sorted((*key, ek), key=lambda b: b._index))
                )
                newpoly[nk] = newpoly.get(nk, 0.0) + coef * float(ck)
        poly = newpoly

    cont = cont_factors[0] if cont_factors else None
    out: Optional[Expression] = None
    for key, coef in poly.items():
        c = const * coef
        if c == 0.0:
            continue
        # Lift the pure-binary monomial ``prod(key)`` to a single 0/1 quantity — the
        # bit itself, or a binary AND aux; both are ``Variable`` (needed by big-M).
        if len(key) == 0:
            mono: Optional[Variable] = None  # empty product == 1
        elif len(key) == 1:
            mono = key[0]
        else:
            mono = exp.and_product(list(key))
        if cont is not None:
            cexpr, clo, chi = cont
            if mono is None:
                term_expr: Expression = cexpr  # 1 * continuous
            else:
                term_expr = exp.bigm_product(mono, cexpr, clo, chi)  # continuous * 0/1
                # Issue #721: tie a multi-bit AND coupling back to the continuous
                # factor via RLT (default-OFF), closing the objective-coupling leak.
                if len(key) >= 2:
                    exp.add_coupling_rlt(term_expr, list(key), cexpr, clo, chi)
        else:
            if mono is None:
                out = Constant(c) if out is None else BinaryOp("+", out, Constant(c))
                continue
            term_expr = mono
        term = term_expr if c == 1.0 else BinaryOp("*", Constant(c), term_expr)
        out = term if out is None else BinaryOp("+", out, term)
    return out if out is not None else Constant(0.0)


def _rewrite(expr: Expression, model: Model, exp: _Expander) -> Expression:
    """Recursively rewrite integer-factor bilinear products in *expr*."""
    if isinstance(expr, BinaryOp):
        if expr.op == "**":
            sq = _try_expand_square(expr, exp)
            if sq is not None:
                return sq
        if expr.op == "*":
            if exp.multilinear:
                ml = _try_expand_multilinear(expr, exp)
                if ml is not None:
                    return ml
            replaced = _try_expand_mul(expr, model, exp)
            if replaced is not None:
                return replaced
        left = _rewrite(expr.left, model, exp)
        right = _rewrite(expr.right, model, exp)
        if left is expr.left and right is expr.right:
            return expr
        return BinaryOp(expr.op, left, right)
    # Other node types: rebuild children generically via known attributes.
    for attr in ("operand",):
        child = getattr(expr, attr, None)
        if isinstance(child, Expression):
            new = _rewrite(child, model, exp)
            if new is not child:
                import copy

                clone = copy.copy(expr)
                setattr(clone, attr, new)
                return clone
    for attr in ("args", "terms"):
        seq = getattr(expr, attr, None)
        if isinstance(seq, (list, tuple)):
            new_seq = [_rewrite(c, model, exp) if isinstance(c, Expression) else c for c in seq]
            if any(n is not o for n, o in zip(new_seq, seq)):
                import copy

                clone = copy.copy(expr)
                setattr(clone, attr, type(seq)(new_seq))
                return clone
    return expr


def _for_each_int_bilinear(expr: Expression, implied, fn) -> None:
    """Call ``fn(int_factor_refs)`` for each integer-factor bilinear product in
    *expr*, where ``int_factor_refs`` is the list of ``(var, elem)`` factors that
    are integer-valued (one or both factors)."""
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            factors = _collect_mul_factors(expr)
            refs = [_scalar_var_ref(f) for f in factors if not isinstance(f, Constant)]
            if all(r is not None for r in refs) and len(refs) == 2:
                (v0, e0), (v1, e1) = refs  # type: ignore[misc]
                if not (v0._index == v1._index and e0 == e1):
                    ints = [
                        (vi, eli)
                        for (vi, eli) in ((v0, e0), (v1, e1))
                        if _int_factor_range(vi, eli, implied)
                    ]
                    if ints:
                        fn(ints)
        _for_each_int_bilinear(expr.left, implied, fn)
        _for_each_int_bilinear(expr.right, implied, fn)
        return
    c = getattr(expr, "operand", None)
    if isinstance(c, Expression):
        _for_each_int_bilinear(c, implied, fn)
    for attr in ("args", "terms"):
        seq = getattr(expr, attr, None)
        if isinstance(seq, (list, tuple)):
            for c in seq:
                if isinstance(c, Expression):
                    _for_each_int_bilinear(c, implied, fn)


def _bodies(model: Model):
    for c in model._constraints:
        if isinstance(c, Constraint):
            yield distribute_products(c.body)
    if model._objective is not None:
        yield distribute_products(model._objective.expression)


def _has_int_square(expr: Expression, implied) -> bool:
    """True if *expr* contains an integer ``x**2`` (or ``x*x``) term this pass can
    exactly linearize."""
    if isinstance(expr, BinaryOp):
        if (
            expr.op == "**"
            and isinstance(expr.right, Constant)
            and abs(float(expr.right.value) - 2.0) < 1e-12
        ):
            ref = _scalar_var_ref(expr.left)
            if ref is not None and _int_factor_range(ref[0], ref[1], implied):
                return True
        if expr.op == "*":
            refs = [
                _scalar_var_ref(f)
                for f in _collect_mul_factors(expr)
                if not isinstance(f, Constant)
            ]
            if len(refs) == 2 and all(r is not None for r in refs):
                (v0, e0), (v1, e1) = refs  # type: ignore[misc]
                if v0._index == v1._index and e0 == e1 and _int_factor_range(v0, e0, implied):
                    return True
        return _has_int_square(expr.left, implied) or _has_int_square(expr.right, implied)
    c = getattr(expr, "operand", None)
    if isinstance(c, Expression) and _has_int_square(c, implied):
        return True
    for attr in ("args", "terms"):
        seq = getattr(expr, attr, None)
        if isinstance(seq, (list, tuple)):
            if any(isinstance(x, Expression) and _has_int_square(x, implied) for x in seq):
                return True
    return False


def _has_int_multilinear(expr: Expression, implied) -> bool:
    """True if *expr* contains an *integer-multilinear* product this pass can
    exactly linearize: a ``*`` term with >=3 scalar-variable factors, at least one
    integer-valued (declared or *implied*) and at most one continuous factor. (Two
    continuous factors is a genuine continuous nonlinearity; fewer than three
    variable factors is the bilinear/square case handled separately.)"""
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            refs = [
                _scalar_var_ref(f)
                for f in _collect_mul_factors(expr)
                if not isinstance(f, Constant)
            ]
            if len(refs) >= 3 and all(r is not None for r in refs):
                n_int = sum(1 for (vi, eli) in refs if _int_factor_range(vi, eli, implied))  # type: ignore[misc]
                n_cont = len(refs) - n_int
                if n_int >= 1 and n_cont <= 1:
                    return True
        return _has_int_multilinear(expr.left, implied) or _has_int_multilinear(expr.right, implied)
    c = getattr(expr, "operand", None)
    if isinstance(c, Expression) and _has_int_multilinear(c, implied):
        return True
    for attr in ("args", "terms"):
        seq = getattr(expr, attr, None)
        if isinstance(seq, (list, tuple)):
            if any(isinstance(x, Expression) and _has_int_multilinear(x, implied) for x in seq):
                return True
    return False


def has_integer_product_work(model: Model, implied=frozenset(), multilinear: bool = False) -> bool:
    """True if any constraint/objective has an integer-factor bilinear product or
    integer square (integer or *implied*-integer factor) this pass can linearize.
    When *multilinear* is set, also detect integer-multilinear products (issue
    #707)."""
    try:
        for body in _bodies(model):
            found = []
            _for_each_int_bilinear(body, implied, lambda ints: found.append(True))
            if found or _has_int_square(body, implied):
                return True
            if multilinear and _has_int_multilinear(body, implied):
                return True
    except Exception:
        return False
    return False


def has_integer_multilinear_work(model: Model, implied=frozenset()) -> bool:
    """True if any constraint/objective has an integer-*multilinear* product
    (>=3 factors, <=1 continuous, >=1 integer/implied-integer) — the class the
    ``DISCOPT_INTEGER_MULTILINEAR_REFORM`` pass linearizes (issue #707)."""
    try:
        return any(_has_int_multilinear(body, implied) for body in _bodies(model))
    except Exception:
        return False


def has_nonconvex_integer_bilinear(model: Model) -> bool:
    """True if the model contains a *distinct-variable* integer-factor bilinear
    product ``x_i*x_j`` (``i != j``). Such a term has an indefinite Hessian, so it
    is a cheap, sound **nonconvexity** witness — and it is exactly the structure
    whose loose McCormick relaxation this pass fixes. Used to gate the
    reformulation to nonconvex models (a convex MIQP with only ``x**2`` squares
    has no distinct bilinear term and is left to the convex fast paths) without
    paying for a full convexity classification.
    """
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        found: list[bool] = []
        for body in _bodies(model):
            _for_each_int_bilinear(body, implied, lambda ints: found.append(True))
            if found:
                return True
    except Exception:
        return False
    return False


def _participation(model: Model, implied) -> dict:
    """Count, per integer factor ``(var._index, elem)``, the number of bilinear
    products it appears in — used to pick the most-shared factor to expand."""
    counts: dict[tuple[int, int], int] = {}

    def tally(ints):
        for vi, eli in ints:
            counts[(vi._index, eli)] = counts.get((vi._index, eli), 0) + 1

    for body in _bodies(model):
        _for_each_int_bilinear(body, implied, tally)
    return counts


def expand_integer_products(model: Model, implied=frozenset(), multilinear: bool = False) -> Model:
    """Return a model equivalent to *model* with integer-factor bilinear products
    binary-expanded + big-M-linearized to their exact (pure-MILP) form. ``implied``
    is an optional set of ``(var._index, elem)`` treated as integer-valued in
    addition to declared integers. When *multilinear* is set, also exact-linearize
    integer-multilinear products (>=3 factors, <=1 continuous — issue #707).
    Returns *model* unchanged when no such product exists or on any unexpected error
    (never regresses)."""
    try:
        if not has_integer_product_work(model, implied, multilinear=multilinear):
            return model
        new_model = Model(model.name)
        new_model._variables = list(model._variables)
        new_model._parameters = list(model._parameters)
        new_model._rebuild_name_index()  # keep the name cache in sync (M7)
        new_model._objective = model._objective
        exp = _Expander(
            new_model,
            implied=implied,
            participation=_participation(model, implied),
            multilinear=multilinear,
        )

        rebuilt: list[Constraint] = []
        for c in model._constraints:
            if not isinstance(c, Constraint):
                rebuilt.append(c)
                continue
            body = _rewrite(distribute_products(c.body), new_model, exp)
            rebuilt.append(c if body is c.body else Constraint(body, c.sense, c.rhs, c.name))

        if new_model._objective is not None:
            obj = new_model._objective
            new_expr = _rewrite(distribute_products(obj.expression), new_model, exp)
            if new_expr is not obj.expression:
                import copy

                new_obj = copy.copy(obj)
                new_obj.expression = new_expr
                new_model._objective = new_obj

        # Blowup guard. ``distribute_products`` can expand a handful of product
        # terms into a combinatorial number of monomials (nvs17: 7 vars -> 2751),
        # each spawning a big-M aux. The result is a valid pure MILP but far too
        # large to solve, so discard it and keep the original model (the solver
        # falls back to the normal nonconvex path — no regression). Legitimate
        # reforms stay well under the cap (ex126x ~2x, nvs14 ~20x at 160 vars).
        cap = max(1000, 6 * len(model._variables))
        if len(new_model._variables) > cap:
            return model
        new_model._constraints = rebuilt + exp.aux_constraints
        _attach_warm_start_spec(model, new_model, exp)
        # #732 Stage 2: configuration metadata for the disjunctive config bound
        # (flat indices over the ORIGINAL columns — the aux block is a suffix, so
        # original offsets are unchanged by the reform). Empty sets when the
        # model has no multilinear configuration structure.
        new_model._ipx_config_indicators = frozenset(exp.config_indicator_flats)
        new_model._ipx_config_counts = frozenset(exp.config_count_flats)
        return new_model
    except Exception:
        return model


def _attach_warm_start_spec(model: Model, new_model: Model, exp: "_Expander") -> None:
    """Record on *new_model* how to reconstruct every auxiliary column from a
    point over the ORIGINAL variables, mirroring the binary-multilinear reform's
    ``_bml_aux_spec`` / ``_bml_n_orig_flat``. The aux columns are appended after
    the originals, so the reformed flat layout is ``[originals | aux]`` and each
    aux's flat index equals ``n_orig_flat + its append position``. Purely primal
    metadata: the MILP driver re-validates any seed built from it, so this can
    never affect the dual bound or the certificate."""
    if not exp.spec_ok:
        return
    full_off: dict[int, int] = {}
    off = 0
    for v in new_model._variables:
        full_off[v._index] = off
        off += v.size
    n_orig_flat = sum(v.size for v in model._variables)

    def _flat(key: tuple[int, int]) -> int:
        return full_off[key[0]] + key[1]

    translated: list[tuple] = []
    try:
        for entry in exp.aux_spec:
            if entry[0] == "bit":
                _, src, k, lo, nbits = entry
                translated.append(("bit", _flat(src), k, lo, nbits))
            elif entry[0] == "and":
                _, z_key, in_keys = entry
                translated.append(("and", _flat(z_key), [_flat(kk) for kk in in_keys]))
            else:  # "prod"
                _, e_key, o_key = entry
                translated.append(("prod", _flat(e_key), _flat(o_key)))
    except KeyError:  # pragma: no cover - defensive; every key is a live variable
        return
    setattr(new_model, "_ipx_aux_spec", translated)  # noqa: B010
    setattr(new_model, "_ipx_n_orig_flat", n_orig_flat)  # noqa: B010
    # Stash the pre-lift model so a primal reseed (the #280 one-hot swap) can run
    # assignment-aware moves over the ORIGINAL variables and map the result back
    # through the spec above. Purely primal metadata; never a soundness lever.
    setattr(new_model, "_ipx_source_model", model)  # noqa: B010


def extend_initial_point(reformed: Model, x0) -> Optional[np.ndarray]:
    """Extend a point over the ORIGINAL variables (flat, length
    ``_ipx_n_orig_flat``) to a point over the reformed model's full variable
    vector by evaluating each auxiliary from its recorded definition
    (``e_k`` = bit k of ``round(x - lo)``; ``v = e * other``). Returns ``None``
    when the model carries no reformulation metadata, the point has the wrong
    length or is non-finite, or any entry cannot be reconstructed exactly (a
    non-integer integer-factor value, or one outside its expansion range) — the
    caller then simply does not seed. The MILP driver re-validates whatever is
    passed (bounds, integrality, row feasibility) and recomputes its objective,
    so this mapping is an optimization, never a soundness lever."""
    spec = getattr(reformed, "_ipx_aux_spec", None)
    n0 = getattr(reformed, "_ipx_n_orig_flat", None)
    if spec is None or n0 is None:
        return None
    x = np.asarray(x0, dtype=np.float64).ravel()
    if x.size != n0 or not np.all(np.isfinite(x)):
        return None
    out: list[float] = list(x)
    for entry in spec:
        if entry[0] == "bit":
            _, src, k, lo, nbits = entry
            n = float(x[src]) - lo
            nr = int(round(n))
            # The bit is exactly determined only when x - lo is a non-negative
            # integer representable in this factor's bit width; anything else
            # (fractional or out-of-range seed) cannot be reconstructed — refuse.
            if abs(n - nr) > 1e-6 or nr < 0 or nr > (1 << nbits) - 1:
                return None
            out.append(float((nr >> k) & 1))
        elif entry[0] == "and":  # z = prod of already-reconstructed input bits
            _, _zi, in_idxs = entry
            prod = 1.0
            for ii in in_idxs:
                prod *= float(out[ii])
            out.append(prod)
        else:  # "prod": v = e * other, both already reconstructed above
            _, ei, oi = entry
            out.append(float(out[ei] * out[oi]))
    return np.asarray(out, dtype=np.float64)


def has_reformulation_work(model: Model) -> bool:
    """True if the model has any integer-factor bilinear product — counting both
    declared-integer and *implied*-integer factors — that this pass can linearize."""
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        return has_integer_product_work(model, implied)
    except Exception:
        return False


def reformulate_integer_bilinear(model: Model) -> Model:
    """End-to-end pass: detect implied-integer factor variables, then exactly
    linearize every integer-factor bilinear product into pure-MILP form. This is
    the entry the solver calls; it is a no-op (returns *model*) when nothing
    applies and never mutates the input model's variable types."""
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        return expand_integer_products(model, implied)
    except Exception:
        return model


def has_integer_multilinear_reformulation_work(model: Model) -> bool:
    """True if the model has an integer-*multilinear* product (>=3 factors, <=1
    continuous, >=1 declared/implied-integer) that the ``#707`` pass linearizes."""
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        return has_integer_multilinear_work(model, implied)
    except Exception:
        return False


def reformulate_integer_multilinear(model: Model) -> Model:
    """End-to-end pass (issue #707): detect implied-integer factors, then exactly
    linearize every integer-factor bilinear **and** integer-multilinear product
    (>=3 factors, <=1 continuous). A no-op (returns *model*) when nothing applies.
    Gated by ``DISCOPT_INTEGER_MULTILINEAR_REFORM``; the caller decides whether the
    result is adopted (pure-MILP → MILP engine, else kept on the spatial path)."""
    try:
        from .implied_integer import detect_implied_integers

        implied = frozenset(detect_implied_integers(model))
        return expand_integer_products(model, implied, multilinear=True)
    except Exception:
        return model
