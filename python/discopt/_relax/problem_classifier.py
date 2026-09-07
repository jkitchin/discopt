"""
Problem classification and standard-form extraction for LP, QP, QCP, MILP,
MIQP, MIQCP, NLP, MINLP.

Uses existing Rust structure detection (is_linear, is_quadratic) via PyO3 bindings
to classify problems, then extracts standard-form data using the JAX DAG compiler.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    # Annotation-only: ``LPData``/``QPData`` are typed as jnp arrays to match the
    # JAX differentiation consumers (``differentiable_qp`` etc.); at runtime they hold numpy from
    # the JAX-free extractors. ``from __future__ import annotations`` keeps these
    # strings, so no JAX import happens at module load.
    import jax.numpy as jnp

# NOTE: ``jax`` is imported lazily inside the ``extract_*`` data functions that
# build jnp arrays. ``classify_problem`` and the dataclasses are purely
# structural (annotations are strings under ``from __future__ import
# annotations``), so importing this module — done on every ``Model.solve`` to
# route the problem class — does not pull in JAX. That keeps LP/MILP/MIQP solves
# free of JAX/XLA cold-start.
from discopt._flat_index import resolve_scalar_slot
from discopt._relax.scalarize import sum_is_full_reduction
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    IndexExpression,
    MatMulExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

logger = logging.getLogger(__name__)


class ProblemClass(Enum):
    """Classification of an optimization problem."""

    LP = "lp"  # linear obj + linear constraints + all continuous
    QP = "qp"  # ≤quadratic obj + linear constraints + all continuous
    QCP = "qcp"  # linear obj + at least one quadratic constraint + all continuous
    QCQP = "qcqp"  # ≤quadratic obj + at least one quadratic constraint + all continuous
    MILP = "milp"  # linear obj + linear constraints + has integer/binary
    MIQP = "miqp"  # ≤quadratic obj + linear constraints + has integer/binary
    MIQCP = "miqcp"  # linear obj + at least one quadratic constraint + has integer/binary
    MIQCQP = "miqcqp"  # ≤quadratic obj + quadratic constraints + has integer/binary
    NLP = "nlp"  # general nonlinear + all continuous
    MINLP = "minlp"  # general nonlinear + has integer/binary


def classify_problem(model: Model) -> ProblemClass:
    """Classify a model into LP, QP, QCP, MILP, MIQP, MIQCP, NLP, or MINLP.

    Uses Rust structure detection for degree analysis of the objective
    and constraints. Falls back to NLP/MINLP if Rust bindings unavailable.

    Args:
        model: A discopt Model with objective and constraints.

    Returns:
        ProblemClass enum value.
    """
    has_integer = any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)

    try:
        from discopt._rust import model_to_repr

        _builder = getattr(model, "_builder", None)
        repr = model_to_repr(model, _builder)
        obj_linear = repr.is_objective_linear()
        obj_quadratic = repr.is_objective_quadratic()
        all_constraints_linear = all(
            repr.is_constraint_linear(i) for i in range(repr.n_constraints)
        )
        if hasattr(repr, "is_constraint_quadratic"):
            all_constraints_quadratic = all(
                repr.is_constraint_quadratic(i) for i in range(repr.n_constraints)
            )
        else:
            all_constraints_quadratic = all_constraints_linear
    except Exception as exc:  # noqa: BLE001 - NLP/MINLP is the always-valid fallback class
        # Not merely an optimization: misrouting an LP/QP to the MINLP path is the
        # difference between the fast family and full spatial B&B, so a silent
        # degradation here reads as "the fast family didn't trigger".
        logger.debug(
            "Rust structure detection unavailable, classifying as NLP/MINLP: %s: %s",
            type(exc).__name__,
            exc,
        )
        return ProblemClass.MINLP if has_integer else ProblemClass.NLP

    if all_constraints_linear:
        if obj_linear:
            return ProblemClass.MILP if has_integer else ProblemClass.LP
        if obj_quadratic:
            return ProblemClass.MIQP if has_integer else ProblemClass.QP

    if all_constraints_quadratic:
        if obj_linear:
            return ProblemClass.MIQCP if has_integer else ProblemClass.QCP
        if obj_quadratic:
            return ProblemClass.MIQCQP if has_integer else ProblemClass.QCQP

    return ProblemClass.MINLP if has_integer else ProblemClass.NLP


# Dense-Q budget for QP extraction (#863). Above this the extractor emits a sparse
# Q instead: a wide model with a narrow objective (watercontamination0202 is 106,711
# variables whose objective touches 101) cannot hold (n, n) float64 at all -- that
# is 91 GB. 256 MB corresponds to n ~= 5,657.
_QP_DENSE_Q_MAX_BYTES = 256 * 1024 * 1024

# Dense-constraint-matrix budget, the exact counterpart for A_eq / A_ub (#863).
# ``watercontamination0202`` is 107,209 rows x 106,711 columns; (m, n) float64 there
# is 91.5 GB, an equal-sized wall to the 91 GB dense Q above. Above this budget the
# extractors emit scipy CSR and consumers densify through ``dense_A()``.
_DENSE_A_MAX_BYTES = 256 * 1024 * 1024

# The quadratic/linear coefficients are read off the expression arena, always.
#
# There used to be a numeric probe here, and a wall-clock budget to stop it from
# eating a whole solve. Both are gone. The probe recovered a Hessian by finite
# differences -- one full model evaluation per SUPPORT PAIR, O(|support|^2) -- for
# a function whose coefficients the arena already holds exactly. Measured over the
# 150-instance MINLPLib MIQP family, probing cost 71,330 s against 5.99 s for the
# walk (up to 28,288 s on ``unitcommit_200_100_1_mod_8`` alone), and the walk
# declined on none of them. Extending that census to every readable BQP / IQP /
# MBQP / MIQP / QCP / MIQCP / BQCP instance -- 424 objectives and all 1,407,252 of
# their constraint rows, up to ``acopf_case13659pegase_qcqp`` at n = 199,281 /
# m = 191,097 -- the walk declined on nothing there either: 0 of 424 and 0 of
# 1,407,252. That is what lets a decline here be a loud refusal rather than a
# guess, and it is the number to re-measure before adding an operator the walk
# does not handle.
#
# It was also less accurate, which is the half that actually mattered. The probe
# identities are differences of nearly-equal floats, so they cancel: #866 records a
# sum of squares recovered as ``Q = 0`` and certified as a false optimum, and on
# ``chimera_mis-01`` the probe spent 275 s and then failed its own verification
# (-171.42 against a true -174.09). Budgeting that was a band-aid on an algorithm
# that should not have been running; deleting it is the fix. Where the walk
# declines, the fallback is the AD tape -- also exact -- and never a probe.


def _sp_issparse(x) -> bool:
    import scipy.sparse as _sp

    return bool(_sp.issparse(x))


class LPData(NamedTuple):
    """Standard-form LP data: min c'x + d s.t. A_eq x = b_eq, x_l <= x <= x_u."""

    c: jnp.ndarray  # (n,) objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


class QPData(NamedTuple):
    """Standard-form QP: min 0.5 x'Qx + c'x + d s.t. A_eq x = b_eq, bounds."""

    Q: jnp.ndarray  # (n, n) quadratic objective matrix (symmetric)
    c: jnp.ndarray  # (n,) linear objective coefficients
    A_eq: jnp.ndarray  # (m, n) equality constraint matrix
    b_eq: jnp.ndarray  # (m,) equality RHS
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


def dense_Q(Q) -> np.ndarray:
    """Materialise a ``QPData.Q`` as a dense ``float64`` array.

    ``Q`` may be a dense array or a scipy sparse matrix (#863: a wide model with a
    narrow objective — ``watercontamination0202`` is 106,711 variables whose
    objective touches 101 — cannot hold a dense ``(n, n)`` ``Q``; that is 91 GB).

    Returns numpy, which is what these fields actually hold at runtime (the module
    header notes ``QPData`` is *annotated* for the JAX differentiation consumers but
    populated by the JAX-free extractors). The call sites that feed POUNCE's
    ``solve_qp_kkt`` cast at the boundary rather than have this function lie about
    its type.

    Every consumer must go through this rather than calling ``np.asarray`` on ``Q``
    directly. ``np.asarray`` on a scipy sparse matrix does **not** raise — it
    returns a 0-d object array wrapping the matrix — so a missed call site would
    silently feed garbage into a solver instead of failing loudly. That is the whole
    reason this helper exists.
    """
    try:
        import scipy.sparse as _sp

        if _sp.issparse(Q):
            return np.asarray(Q.toarray(), dtype=np.float64)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    # Inspect BEFORE coercing to float64: np.asarray(obj_array, dtype=float) raises
    # a bare ValueError, which hides what actually went wrong.
    arr = np.asarray(Q)
    if arr.dtype == object or arr.ndim != 2:
        raise TypeError(
            f"QPData.Q densified to dtype={arr.dtype} ndim={arr.ndim}; a sparse "
            "matrix probably reached np.asarray without going through dense_Q()"
        )
    return np.asarray(arr, dtype=np.float64)


def dense_A(A) -> np.ndarray:
    """Materialise a constraint matrix as a dense ``float64`` 2-D array.

    Covers ``LPData.A_eq``, ``QPData.A_eq`` and ``QCPData.A_ub`` / ``A_eq``. Each
    may be a dense array or a scipy sparse matrix (#863). The dense form is
    ``(m, n)``: on ``watercontamination0202`` — 106,711 variables, 107,209 rows —
    that is **91.5 GB**, an equal-sized wall to the 91 GB dense ``Q`` that the same
    issue removed, and it is why extraction on that instance never returns.

    The sibling of :func:`dense_Q`, and for the same reason. ``np.asarray`` on a
    scipy sparse matrix does **not** raise — it returns a 0-d object array wrapping
    the matrix — so a consumer that keeps calling ``np.asarray(lp_data.A_eq)``
    directly would silently feed garbage into a solver instead of failing. Every
    consumer must go through this, and it raises loudly (``TypeError`` naming the
    cause) if it ever yields an object array or a non-2-D result, which is the
    signature of a missed call site.

    Returns numpy, which is what these fields actually hold at runtime (the module
    header notes the data classes are *annotated* for the JAX consumers but
    populated by the JAX-free extractors); the sites that feed POUNCE's
    ``solve_qp_kkt`` cast at that boundary rather than have this function lie
    about its type.
    """
    try:
        import scipy.sparse as _sp

        if _sp.issparse(A):
            return np.asarray(A.toarray(), dtype=np.float64)
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        pass
    # Inspect BEFORE coercing to float64: np.asarray(obj_array, dtype=float) raises
    # a bare ValueError, which hides what actually went wrong.
    arr = np.asarray(A)
    if arr.dtype == object or arr.ndim != 2:
        raise TypeError(
            f"constraint matrix densified to dtype={arr.dtype} ndim={arr.ndim}; a "
            "sparse matrix probably reached np.asarray without going through dense_A()"
        )
    return np.asarray(arr, dtype=np.float64)


class QuadraticConstraintData(NamedTuple):
    """Quadratic row data: 0.5 x'Qx + c'x sense rhs."""

    Q: jnp.ndarray
    c: jnp.ndarray
    sense: str
    rhs: float


class QCPData(NamedTuple):
    """Standard-form QCP/QCQP data with explicit linear and quadratic rows."""

    Q: jnp.ndarray  # (n, n) quadratic objective matrix (symmetric)
    c: jnp.ndarray  # (n,) linear objective coefficients
    A_ub: jnp.ndarray  # (m_ub, n) linear inequality matrix
    b_ub: jnp.ndarray  # (m_ub,) linear inequality RHS
    A_eq: jnp.ndarray  # (m_eq, n) linear equality matrix
    b_eq: jnp.ndarray  # (m_eq,) linear equality RHS
    quadratic_constraints: tuple[QuadraticConstraintData, ...]
    x_l: jnp.ndarray  # (n,) lower bounds
    x_u: jnp.ndarray  # (n,) upper bounds
    obj_const: float = 0.0  # constant term in objective


def _get_variable_bounds(model: Model):
    """Extract flat lower and upper bounds from model variables.

    Returns numpy arrays to avoid JAX device-transfer overhead during
    extraction.  POUNCE's ``solve_lp_kkt``/``solve_qp_kkt`` consume numpy
    directly; only the ``custom_jvp`` differentiation wrappers convert to
    ``jnp.array``, and only for the tangent solve.
    """
    lb_parts = []
    ub_parts = []
    for v in model._variables:
        lb_parts.append(v.lb.flatten())
        ub_parts.append(v.ub.flatten())
    n = sum(v.size for v in model._variables)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    lb = np.concatenate(lb_parts).astype(np.float64)
    ub = np.concatenate(ub_parts).astype(np.float64)
    return lb, ub


# ---------------------------------------------------------------------------
# Algebraic coefficient extraction (no autodiff)
# ---------------------------------------------------------------------------


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Return the starting offset of a variable in the flat x vector.

    Delegates to ``Model._flat_var_offset``, which memoizes an exclusive prefix-sum
    table and rebuilds it only when the (append-only) variable list grows, so each
    lookup is O(1). This function used to re-sum ``model._variables[: var._index]``
    from scratch — O(n_vars) per *variable reference* — which is the very quadratic
    that #654 removed everywhere else and that never got removed here.

    That omission is why extraction on ``watercontamination0202`` never returned.
    It has 106,711 variables and 107,209 constraints; the walk resolves an offset
    per variable reference per row, at a measured mean ``var._index`` of 59,909.
    Measured on that instance, sampling rows with a stride across the whole range
    (sampling only the FIRST rows hides it — they reference the lowest indices,
    where the scan is cheapest):

        rescan (before)      4.108 ms/row  ->  440 s over 107,209 rows
        memoized (after)     0.012 ms/row  ->    1.3 s
                                                340x

    Pure speedup: ``_variables`` only ever grows and a Variable's ``_index`` /
    ``size`` are immutable after construction, so the returned offset is identical
    to the rescan's (see the docstring on ``Model._flat_var_offset``).
    """
    return model._flat_var_offset(var)


class _NotLinearError(Exception):
    """Raised when an expression is not linear."""


class _NotQuadraticError(Exception):
    """Raised when an expression is not quadratic (at most degree 2)."""


def _extract_linear_coefficients(expr, model: Model, n: int):
    """Walk an expression tree to extract linear coefficients and constant.

    Returns (coefficients, constant) where:
      - coefficients is a numpy array of shape (n,) with coefficient for each variable slot
      - constant is a float scalar

    Dense wrapper over :func:`_extract_linear_coefficients_sparse`; bit-identical to
    it, because the dict accumulates exactly the same additions in the same order
    starting from the same 0.0. Callers that assemble many rows should use the sparse
    form directly and never materialise the full-width vector (#863).

    Raises _NotLinearError if the expression is not linear.
    """
    terms, const = _extract_linear_coefficients_sparse(expr, model, n)
    c = np.zeros(n, dtype=np.float64)
    for _i, _v in terms.items():
        c[_i] = _v
    return c, const


def _extract_linear_coefficients_sparse(expr, model: Model, n: int):
    """As :func:`_extract_linear_coefficients`, but keeping the row SPARSE.

    Returns ``(terms, constant)`` where ``terms`` is ``{flat_index: coefficient}``
    in first-touch order. This is what lets ``_extract_constraints_algebraic``
    assemble a (107,209 x 106,711) matrix without a 91.5 GB dense intermediate
    (#863) — the dense full-width row per constraint was never needed, only its
    nonzeros are.

    Raises _NotLinearError if the expression is not linear.
    """
    terms: dict[int, float] = {}
    const = 0.0

    def _add(i: int, v: float) -> None:
        # The dense predecessor got this bound check for free from numpy (and, for a
        # negative index, silently wrote the WRONG slot via numpy wraparound).
        # _NotLinearError rather than IndexError so the dispatcher falls through to
        # the next extractor exactly as it did on the IndexError before.
        if not 0 <= i < n:
            raise _NotLinearError(f"variable slot {i} outside the model's {n} flat slots")
        terms[i] = terms.get(i, 0.0) + v

    def _walk(root, root_scale=1.0, root_allow_array=False):
        # ``allow_array`` is True only inside a ``sum(...)`` reduction, where a
        # size>1 (sub)expression contributes a single scalar row (its element sum
        # with a *uniform* scale). Outside a sum, encountering a size>1 array node
        # in scalar position means the whole body is vector-valued: the algebraic
        # extractor would collapse it to one summed row (C-29), certifying an
        # infeasible point. Refuse instead so extract_lp_data() routes the body to
        # the per-component autodiff extractor (one LP row per element).
        #
        # **Iterative, not recursive (#1064).** A ``.nl`` body is a left-leaning
        # ``((((a+b)+c)+d)+...)`` chain one node deep per term, so a recursive walk
        # needs one Python frame per term and dies on ordinary corpus instances:
        # ``sporttournament40``, ``edgecross24-057`` and 15 others raised
        # ``RecursionError`` here, which the caller reports as "this row is not
        # linear" -- indistinguishable from a genuinely nonlinear row. Children are
        # pushed in reverse so they pop in source order, keeping the floating-point
        # accumulation into ``terms``/``const`` bit-identical (CLAUDE.md §5 regime 1).
        nonlocal const

        stack: list[tuple[object, float, bool]] = [(root, root_scale, root_allow_array)]
        while stack:
            node, scale, allow_array = stack.pop()

            if isinstance(node, Constant):
                val = node.value
                if val.ndim == 0 or val.size == 1:
                    const += scale * float(val.reshape(()))
                else:
                    raise _NotLinearError("Array constant in unexpected position")
                continue

            if isinstance(node, Variable):
                offset = _compute_var_offset(node, model)
                if node.size == 1:
                    _add(offset, scale)
                elif allow_array:
                    # Inside sum(): sum(scale * x) = scale * Σ x_j (uniform scale).
                    for j in range(node.size):
                        _add(offset + j, scale)
                else:
                    raise _NotLinearError(
                        "Array variable in scalar position (vector-valued body); "
                        "routing to the per-component extractor"
                    )
                continue

            if isinstance(node, IndexExpression):
                if isinstance(node.base, Variable):
                    var = node.base
                    offset = _compute_var_offset(var, model)
                    idx = node.index
                    if isinstance(idx, (int, np.integer)):
                        _add(offset + int(idx), scale)
                    elif (
                        isinstance(idx, tuple)
                        and len(idx) == 1
                        and isinstance(idx[0], (int, np.integer))
                    ):
                        _add(offset + int(idx[0]), scale)
                    else:
                        # Multi-dimensional index: flatten
                        try:
                            flat_idx = np.ravel_multi_index(
                                idx if isinstance(idx, tuple) else (idx,), var.shape
                            )
                        except (TypeError, ValueError):
                            # Sliced/partial subscript (vectorized term): this scalar
                            # extractor cannot express it; classify as not-linear.
                            raise _NotLinearError(
                                f"non-scalar index {idx!r} on {var.name}"
                            ) from None
                        _add(offset + int(flat_idx), scale)
                    continue
                raise _NotLinearError(f"IndexExpression on non-variable: {type(node.base)}")

            if isinstance(node, BinaryOp):
                if node.op == "+":
                    stack.append((node.right, scale, allow_array))
                    stack.append((node.left, scale, allow_array))
                    continue
                if node.op == "-":
                    stack.append((node.right, -scale, allow_array))
                    stack.append((node.left, scale, allow_array))
                    continue
                if node.op == "*":
                    # One side must be a scalar constant for linearity. A size>1 array
                    # constant here raises _NotLinearError from _eval_const (the scale
                    # would differ per element), which routes to the per-component
                    # extractor rather than collapsing.
                    if _is_const_expr(node.left):
                        cval = _eval_const(node.left)
                        stack.append((node.right, scale * cval, allow_array))
                        continue
                    if _is_const_expr(node.right):
                        cval = _eval_const(node.right)
                        stack.append((node.left, scale * cval, allow_array))
                        continue
                    raise _NotLinearError("Product of two variable expressions")
                if node.op == "/":
                    if _is_const_expr(node.right):
                        cval = _eval_const(node.right)
                        stack.append((node.left, scale / cval, allow_array))
                        continue
                    raise _NotLinearError("Division by variable expression")
                raise _NotLinearError(f"Non-linear operator: {node.op}")

            if isinstance(node, UnaryOp):
                if node.op == "neg":
                    stack.append((node.operand, -scale, allow_array))
                    continue
                raise _NotLinearError(f"Non-linear unary op: {node.op}")

            if isinstance(node, SumOverExpression):
                for term in reversed(node.terms):
                    stack.append((term, scale, allow_array))
                continue

            if isinstance(node, SumExpression):
                # A FULL reduction sums its operand to a scalar: element-collapse
                # is legitimate here (uniform scale), so allow array nodes beneath
                # this point. An AXIS reduction is array-valued — ``sum(A, axis=1)``
                # is one row per row of ``A`` — so descending would fold rows the
                # model keeps apart and emit ``sum(A) <= b`` for ``sum(A, axis=1)
                # <= b``, a strictly larger feasible set certified as the answer
                # (#1160). Refuse so ``extract_lp_data()`` routes the body to the
                # tape/autodiff extractor, which fans it out into one row per
                # element. Under ``allow_array`` an enclosing reduction already
                # sums every element with this same uniform scale, and summing a
                # partial sum's elements is summing the operand's, so the collapse
                # is exact there and stays allowed.
                if not allow_array and not sum_is_full_reduction(node):
                    raise _NotLinearError(
                        "axis-reduced sum in scalar position (vector-valued body); "
                        "routing to the per-component extractor"
                    )
                stack.append((node.operand, scale, True))
                continue

            if isinstance(node, MatMulExpression):
                # Handle Constant @ Variable or Variable @ Constant
                if isinstance(node.left, Constant) and isinstance(node.right, Variable):
                    mat = node.left.value
                    var = node.right
                    offset = _compute_var_offset(var, model)
                    # mat @ var => result is mat @ x[offset:offset+size]
                    # For 1-D mat (dot product), coefficients are mat elements
                    if mat.ndim == 1:
                        for j in range(var.size):
                            _add(offset + j, scale * float(mat[j]))
                    elif mat.ndim == 2:
                        # Returns vector; this should be used inside a sum
                        raise _NotLinearError("MatMul returning vector in scalar context")
                    continue
                if isinstance(node.right, Constant) and isinstance(node.left, Variable):
                    mat = node.right.value
                    var = node.left
                    offset = _compute_var_offset(var, model)
                    if mat.ndim == 1:
                        for j in range(var.size):
                            _add(offset + j, scale * float(mat[j]))
                        continue
                    raise _NotLinearError("MatMul returning vector in scalar context")
                raise _NotLinearError("MatMul between non-trivial expressions")

            raise _NotLinearError(f"Unhandled expression type: {type(node).__name__}")

    _walk(expr)
    return terms, const


def _materialise_Q(terms: dict[tuple[int, int], float], n: int) -> np.ndarray:
    """Assemble a Hessian from ``{(row, col): value}`` accumulated entries (#863).

    Dense while ``(n, n)`` float64 fits ``_QP_DENSE_Q_MAX_BYTES`` — bit-identical to
    the ``np.zeros((n, n))`` this replaced, because the dict performs the same
    ``+=`` additions in the same order starting from the same 0.0 — and scipy CSR
    beyond it. ``dense_Q()`` re-densifies for consumers.

    The dict is what makes the sparse arm reachable at all. ``np.zeros((n, n))`` is
    91 GB on ``watercontamination0202`` (106,711 variables); macOS *allows* that
    allocation because zero pages are mapped lazily, so it does not raise — it just
    makes the first full read of the array catastrophic. Measured on that instance,
    a single ``Q @ x`` against the lazily-allocated dense Q (holding 4,017 nonzeros)
    took **16.0 s**. Accumulating entries instead keeps peak memory at O(nnz), so
    the dense matrix never has to exist even transiently.

    Mirrors :func:`_materialise_A`, which did the same for the constraint matrix.
    """
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if (n * n * 8) <= _QP_DENSE_Q_MAX_BYTES:
        Q = np.zeros((n, n), dtype=np.float64)
        for (_i, _j), _v in terms.items():
            Q[_i, _j] = _v
        return Q
    import scipy.sparse as _sp

    if not terms:
        # csr_matrix((data, (row, col))) needs non-empty index arrays to infer dtype.
        return cast(np.ndarray, _sp.csr_matrix((n, n), dtype=np.float64))
    r = np.fromiter((k[0] for k in terms), dtype=np.intp, count=len(terms))
    c = np.fromiter((k[1] for k in terms), dtype=np.intp, count=len(terms))
    v = np.fromiter(terms.values(), dtype=np.float64, count=len(terms))
    # Annotated ndarray because that is what every consumer sees after dense_Q();
    # the sparse arm is deliberately outside the annotation, exactly as the repr
    # extractor's producer is (c525f519).
    return cast(np.ndarray, _sp.csr_matrix((v, (r, c)), shape=(n, n)))


def _quadratic_terms_nonempty(terms: dict[tuple[int, int], float], tol: float = 1e-12) -> bool:
    """``_quadratic_row_has_terms`` on the pre-materialisation accumulator.

    Identical predicate (entries absent from the dict are exactly 0.0 in both arms),
    but it never needs the matrix, so a QCP row can be classified linear-or-quadratic
    without materialising anything (#863).
    """
    return any(abs(_v) > tol for _v in terms.values())


def _extract_quadratic_coefficients(expr, model: Model, n: int):
    """Walk expression tree to extract quadratic and linear coefficients.

    Returns (Q, c, constant) where:
      - Q is the Hessian (f = 0.5 x'Qx + c'x + const): a dense (n, n) numpy array
        while that fits ``_QP_DENSE_Q_MAX_BYTES``, scipy CSR beyond it (#863) --
        consumers densify through :func:`dense_Q`
      - c is (n,) numpy array of linear coefficients
      - constant is a float scalar

    Callers that only need the *predicate* "does this have quadratic terms" or the
    nonzero entries should use :func:`_extract_quadratic_terms` and avoid
    materialising anything.

    Raises _NotQuadraticError if the expression has degree > 2.
    """
    terms, c, const = _extract_quadratic_terms(expr, model, n)
    return _materialise_Q(terms, n), c, const


def _extract_quadratic_terms(expr, model: Model, n: int):
    """As :func:`_extract_quadratic_coefficients`, but keeping the Hessian SPARSE.

    Returns ``(terms, c, constant)`` with ``terms`` a ``{(row, col): value}`` dict in
    first-touch order. The full-width ``(n, n)`` Hessian was never needed by the
    walk, only its nonzeros; see :func:`_materialise_Q` for why it must not be
    allocated on a wide model.
    """
    q_terms: dict[tuple[int, int], float] = {}
    c = np.zeros(n, dtype=np.float64)
    const = 0.0

    def _qadd(i: int, j: int, v: float) -> None:
        # The dense predecessor got this bound check for free from numpy (and, for a
        # negative index, silently wrote the WRONG cell via numpy wraparound).
        # _NotQuadraticError rather than IndexError so the dispatcher falls through
        # to the next extractor exactly as it did on the IndexError before.
        if not (0 <= i < n and 0 <= j < n):
            raise _NotQuadraticError(f"Hessian cell ({i}, {j}) outside the model's {n} flat slots")
        q_terms[(i, j)] = q_terms.get((i, j), 0.0) + v

    def _get_var_index(node):
        """Get the flat variable index for a variable-like node, or None."""
        if isinstance(node, Variable):
            if node.size != 1:
                return None
            return _compute_var_offset(node, model)
        if isinstance(node, IndexExpression):
            # #941: the bare-int fast path was `offset + int(idx)`, wrong for a
            # negative index. (The `ravel_multi_index` fallback below it already
            # refused negatives, by raising — so only the fast path was unsound.)
            return resolve_scalar_slot(node, model)
        return None

    def _walk(root, root_scale=1.0, root_allow_array=False):
        # See _extract_linear_coefficients._walk: ``allow_array`` is True only
        # inside a sum() reduction, where a size>1 array variable legitimately
        # collapses to a single scalar term. Outside a sum, an array variable in
        # scalar position means a vector-valued body that must NOT be collapsed to
        # one row (C-29) — refuse so the caller routes to the autodiff extractor.
        #
        # **Iterative, not recursive (#1064).** A ``.nl`` objective is a
        # left-leaning ``((((a+b)+c)+d)+...)`` chain one node deep per term, so a
        # recursive walk needs one Python frame per term and dies on the ordinary
        # case: ``squfl025-040`` (1000 terms) and ``squfl015-080`` (1200) both
        # raised ``RecursionError`` here, which ``_extract_quadratic_coefficients``
        # reports as "not quadratic". A genuinely convex separable MIQP therefore
        # failed to be *recognised* as one — no Hessian, no convex route, no
        # perspective structure — purely because its objective was long. The work
        # stack removes the depth limit; nothing else about the walk changes.
        #
        # Children are pushed in reverse so they pop in source order: the
        # accumulations into ``c``/``const``/``q_terms`` are floating-point sums,
        # and reordering them would perturb the last ulp of every extracted
        # coefficient — a bound-neutral change under CLAUDE.md §5 regime 1 must be
        # bit-identical, so the order is preserved deliberately.
        nonlocal const

        stack: list[tuple[object, float, bool]] = [(root, root_scale, root_allow_array)]
        while stack:
            node, scale, allow_array = stack.pop()

            if isinstance(node, Constant):
                val = node.value
                if val.ndim == 0 or val.size == 1:
                    const += scale * float(val.reshape(()))
                else:
                    raise _NotQuadraticError("Array constant in unexpected position")
                continue

            if isinstance(node, (Variable, IndexExpression)):
                idx = _get_var_index(node)
                if idx is not None:
                    c[idx] += scale
                    continue
                if isinstance(node, Variable) and node.size > 1:
                    if not allow_array:
                        raise _NotQuadraticError(
                            "Array variable in scalar position (vector-valued body); "
                            "routing to the per-component extractor"
                        )
                    offset = _compute_var_offset(node, model)
                    for j in range(node.size):
                        c[offset + j] += scale
                    continue
                raise _NotQuadraticError(f"Cannot extract index from {node}")

            if isinstance(node, BinaryOp):
                if node.op == "+":
                    stack.append((node.right, scale, allow_array))
                    stack.append((node.left, scale, allow_array))
                    continue
                if node.op == "-":
                    stack.append((node.right, -scale, allow_array))
                    stack.append((node.left, scale, allow_array))
                    continue
                if node.op == "*":
                    # Check: const * expr, expr * const, or var * var
                    if _is_const_expr(node.left):
                        cval = _eval_const(node.left)
                        stack.append((node.right, scale * cval, allow_array))
                        continue
                    if _is_const_expr(node.right):
                        cval = _eval_const(node.right)
                        stack.append((node.left, scale * cval, allow_array))
                        continue
                    # var * var => quadratic term
                    # Q is the Hessian: f = 0.5 x'Qx, so d²(c*xi*xj)/dxi dxj = c,
                    # but d²(c*xi²)/dxi² = 2c. We store the Hessian directly.
                    idx_l = _get_var_index(node.left)
                    idx_r = _get_var_index(node.right)
                    if idx_l is not None and idx_r is not None:
                        if idx_l == idx_r:
                            _qadd(idx_l, idx_r, 2.0 * scale)
                        else:
                            _qadd(idx_l, idx_r, scale)
                            _qadd(idx_r, idx_l, scale)
                        continue
                    # Handle (const * var) * var or var * (const * var):
                    # e.g., (Q[i,j] * x[i]) * x[j] from left-to-right evaluation
                    cv_l = _try_extract_const_var(node.left, model)
                    if cv_l is not None and idx_r is not None:
                        cval, idx_l2 = cv_l
                        if idx_l2 == idx_r:
                            _qadd(idx_l2, idx_r, 2.0 * scale * cval)
                        else:
                            _qadd(idx_l2, idx_r, scale * cval)
                            _qadd(idx_r, idx_l2, scale * cval)
                        continue
                    cv_r = _try_extract_const_var(node.right, model)
                    if cv_r is not None and idx_l is not None:
                        cval, idx_r2 = cv_r
                        if idx_l == idx_r2:
                            _qadd(idx_l, idx_r2, 2.0 * scale * cval)
                        else:
                            _qadd(idx_l, idx_r2, scale * cval)
                            _qadd(idx_r2, idx_l, scale * cval)
                        continue
                    raise _NotQuadraticError("Product of non-simple variable expressions")
                if node.op == "/":
                    if _is_const_expr(node.right):
                        cval = _eval_const(node.right)
                        stack.append((node.left, scale / cval, allow_array))
                        continue
                    raise _NotQuadraticError("Division by variable expression")
                if node.op == "**":
                    # x**2 => quadratic
                    if _is_const_expr(node.right):
                        pval = _eval_const(node.right)
                        if abs(pval - 2.0) < 1e-12:
                            idx = _get_var_index(node.left)
                            if idx is not None:
                                _qadd(idx, idx, 2.0 * scale)  # x^2 = 0.5 * 2 * x^2
                                continue
                        if abs(pval - 1.0) < 1e-12:
                            stack.append((node.left, scale, allow_array))
                            continue
                        if abs(pval) < 1e-12:
                            const += scale
                            continue
                    raise _NotQuadraticError(f"Power with exponent {node.right}")
                raise _NotQuadraticError(f"Unknown binary op: {node.op}")

            if isinstance(node, UnaryOp):
                if node.op == "neg":
                    stack.append((node.operand, -scale, allow_array))
                    continue
                raise _NotQuadraticError(f"Non-linear unary op: {node.op}")

            if isinstance(node, SumOverExpression):
                for term in reversed(node.terms):
                    stack.append((term, scale, allow_array))
                continue

            if isinstance(node, SumExpression):
                # Full reduction: array collapse legitimate below here. Axis
                # reduction: array-valued, one row per surviving element, so
                # descending would fold separate rows into one (#1160) — see the
                # matching guard in ``_extract_linear_coefficients_sparse``.
                if not allow_array and not sum_is_full_reduction(node):
                    raise _NotQuadraticError(
                        "axis-reduced sum in scalar position (vector-valued body); "
                        "routing to the per-component extractor"
                    )
                stack.append((node.operand, scale, True))
                continue

            if isinstance(node, MatMulExpression):
                # Handle Constant @ Variable for linear parts of QP constraints
                if isinstance(node.left, Constant) and isinstance(node.right, Variable):
                    mat = node.left.value
                    var = node.right
                    offset = _compute_var_offset(var, model)
                    if mat.ndim == 1:
                        for j in range(var.size):
                            c[offset + j] += scale * float(mat[j])
                        continue
                    raise _NotQuadraticError("MatMul returning vector")
                if isinstance(node.right, Constant) and isinstance(node.left, Variable):
                    mat = node.right.value
                    var = node.left
                    offset = _compute_var_offset(var, model)
                    if mat.ndim == 1:
                        for j in range(var.size):
                            c[offset + j] += scale * float(mat[j])
                        continue
                    raise _NotQuadraticError("MatMul returning vector")
                raise _NotQuadraticError("MatMul between non-trivial expressions")

            raise _NotQuadraticError(f"Unhandled expression type: {type(node).__name__}")

    _walk(expr)
    return q_terms, c, const


def _try_extract_const_var(expr, model: Model):
    """Try to decompose expr as (constant * variable).

    Returns (constant_value, flat_var_index) if expr is of the form
    Constant * Variable/IndexExpr or Variable/IndexExpr * Constant,
    or just a bare Variable/IndexExpr (constant = 1.0).

    Returns None if the expression is not of this form.
    """
    # Bare variable => coefficient 1.0
    if isinstance(expr, (Variable, IndexExpression)):
        if isinstance(expr, Variable) and expr.size != 1:
            return None
        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            offset = _compute_var_offset(expr.base, model)
            idx = expr.index
            if isinstance(idx, (int, np.integer)):
                return (1.0, offset + int(idx))
            if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], (int, np.integer)):
                return (1.0, offset + int(idx[0]))
            try:
                flat_idx = np.ravel_multi_index(
                    idx if isinstance(idx, tuple) else (idx,), expr.base.shape
                )
            except (TypeError, ValueError):
                return None  # sliced/partial subscript: not a scalar reference
            return (1.0, offset + int(flat_idx))
        if isinstance(expr, Variable):
            return (1.0, _compute_var_offset(expr, model))
        return None

    # const * var or var * const
    if isinstance(expr, BinaryOp) and expr.op == "*":
        if _is_const_expr(expr.left):
            cval = _eval_const(expr.left)
            inner = _try_extract_const_var(expr.right, model)
            if inner is not None:
                return (cval * inner[0], inner[1])
        if _is_const_expr(expr.right):
            cval = _eval_const(expr.right)
            inner = _try_extract_const_var(expr.left, model)
            if inner is not None:
                return (cval * inner[0], inner[1])

    # neg(var) => -1.0 * var
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _try_extract_const_var(expr.operand, model)
        if inner is not None:
            return (-inner[0], inner[1])

    return None


def _is_const_expr(expr) -> bool:
    """Check if an expression is a pure constant (no variables)."""
    if isinstance(expr, Constant):
        return True
    if isinstance(expr, (Variable, IndexExpression)):
        return False
    if isinstance(expr, BinaryOp):
        return _is_const_expr(expr.left) and _is_const_expr(expr.right)
    if isinstance(expr, UnaryOp):
        return _is_const_expr(expr.operand)
    if isinstance(expr, SumOverExpression):
        return all(_is_const_expr(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _is_const_expr(expr.operand)
    return False


def _eval_const(expr) -> float:  # type: ignore[return-value]
    """Evaluate a constant expression to a float scalar.

    Raises ``_NotLinearError`` (NOT ``ValueError``) on a non-scalar array
    constant. C-30: a raw ``ValueError`` from ``float(v.item())`` on a size>1
    array (e.g. ``sum(np.array([1,1]) * x)``) used to abort the algebraic walk
    and mis-route to a fallback that dropped the objective sense. Refusing with
    ``_NotLinearError`` routes such bodies to the row- and sense-correct autodiff
    extractor instead.
    """
    if isinstance(expr, Constant):
        v = expr.value
        if v.ndim == 0 or v.size == 1:
            return float(v.reshape(()))
        raise _NotLinearError(
            "Non-scalar array constant cannot be evaluated as a scalar coefficient; "
            "routing to the per-component extractor"
        )
    if isinstance(expr, BinaryOp):
        lv = _eval_const(expr.left)
        r = _eval_const(expr.right)
        if expr.op == "+":
            return lv + r
        if expr.op == "-":
            return lv - r
        if expr.op == "*":
            return lv * r
        if expr.op == "/":
            return lv / r
        if expr.op == "**":
            return float(lv**r)
        raise ValueError(f"Unknown op in const eval: {expr.op}")
    if isinstance(expr, UnaryOp):
        uv = _eval_const(expr.operand)
        if expr.op == "neg":
            return float(-uv)
        if expr.op == "abs":
            return float(abs(uv))
        raise ValueError(f"Unknown unary op in const eval: {expr.op}")
    if isinstance(expr, SumOverExpression):
        return sum(_eval_const(t) for t in expr.terms)
    if isinstance(expr, SumExpression):
        return _eval_const(expr.operand)
    raise ValueError(f"Not a constant expression: {type(expr).__name__}")


def _extract_constraints_algebraic(model: Model, n_orig: int):
    """Extract linear constraint data algebraically (shared by LP and QP paths).

    Returns (A_eq, b_eq, x_l, x_u, n_slack) where slacks are appended for
    inequality constraints. ``A_eq`` is dense while it fits ``_DENSE_A_MAX_BYTES``
    and scipy CSR beyond it (#863) — consumers densify via ``dense_A()``.

    Raises _NotLinearError if any constraint is not linear.
    """
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    eq_terms: list[dict[int, float]] = []
    eq_rhs: list[float] = []
    ineq_terms: list[dict[int, float]] = []
    ineq_senses: list[str] = []
    ineq_rhs: list[float] = []

    for con in constraints:
        # Sparse walk: the dense full-width row per constraint was never needed,
        # only its nonzeros (#863).
        terms, const = _extract_linear_coefficients_sparse(con.body, model, n_orig)
        if con.sense == "==":
            eq_terms.append(terms)
            eq_rhs.append(-const)
        elif con.sense == "<=":
            ineq_terms.append(terms)
            ineq_senses.append("le")
            ineq_rhs.append(-const)
        elif con.sense == ">=":
            ineq_terms.append(terms)
            ineq_senses.append("ge")
            ineq_rhs.append(-const)

    n_eq = len(eq_terms)
    n_ineq = len(ineq_terms)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    # COO triples, not np.stack() of dense (n_total,) rows: that stack is
    # 107,209 x 106,711 float64 = 91.5 GB on watercontamination0202, and it needs
    # every row resident simultaneously (#863).
    coo_rows: list[int] = []
    coo_cols: list[int] = []
    coo_vals: list[float] = []
    b_vals: list[float] = []

    for i in range(n_eq):
        _append_row_coo(coo_rows, coo_cols, coo_vals, i, eq_terms[i])
        b_vals.append(eq_rhs[i])

    for i in range(n_ineq):
        r = n_eq + i
        _append_row_coo(coo_rows, coo_cols, coo_vals, r, ineq_terms[i])
        # body <= 0 becomes body + s = 0; body >= 0 becomes body - s = 0; s >= 0.
        coo_rows.append(r)
        coo_cols.append(n_orig + i)
        coo_vals.append(1.0 if ineq_senses[i] == "le" else -1.0)
        b_vals.append(ineq_rhs[i])

    m_total = n_eq + n_ineq
    A_eq = _materialise_A(coo_rows, coo_cols, coo_vals, m_total, n_total)
    b_eq = np.array(b_vals, dtype=np.float64)

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, 1e20, dtype=np.float64)])

    return A_eq, b_eq, x_l, x_u, n_slack


def _quadratic_row_has_terms(Q: np.ndarray, tol: float = 1e-12) -> bool:
    """True when ``Q`` holds a nonzero entry above ``tol``.

    Sparse-aware (#875): ``np.abs`` on a scipy sparse matrix returns a sparse matrix,
    and ``np.any`` on one does not mean what it does on an ndarray — the stored
    values are the only candidates, so test those directly. Explicit zeros can be
    stored, hence the ``> tol`` test rather than ``nnz``.
    """
    if _sp_issparse(Q):
        return bool(np.any(np.abs(Q.data) > tol))
    return bool(np.any(np.abs(Q) > tol))


def _empty_matrix(n_cols: int) -> np.ndarray:
    return np.zeros((0, n_cols), dtype=np.float64)


def _materialise_A(
    rows: list[int], cols: list[int], vals: list[float], m: int, n: int
) -> np.ndarray:
    """Assemble a constraint matrix from COO triples (#863).

    Dense while ``(m, n)`` float64 fits ``_DENSE_A_MAX_BYTES`` — bit-identical to the
    ``np.stack`` of dense full-width rows this replaced — and scipy CSR beyond it,
    because a model with both many rows and many columns cannot hold ``(m, n)``
    floats at all: ``watercontamination0202`` is 107,209 x 106,711, which is 91.5 GB.
    ``dense_A()`` re-densifies for consumers.

    The COO form is what makes the sparse arm reachable at all. ``np.stack`` needs
    every dense row resident simultaneously; contributing rows as triples keeps peak
    memory at O(nnz), so the dense matrix never has to exist even transiently.

    Entries absent from the triples are 0.0 in both arms, so dropping explicitly
    zero coefficients (as ``_append_dense_row_coo`` does) cannot change the
    densified result.
    """
    if m == 0:
        return _empty_matrix(n)
    r = np.asarray(rows, dtype=np.intp)
    c = np.asarray(cols, dtype=np.intp)
    v = np.asarray(vals, dtype=np.float64)
    if (m * n * 8) <= _DENSE_A_MAX_BYTES:
        A = np.zeros((m, n), dtype=np.float64)
        A[r, c] = v
        return A
    import scipy.sparse as _sp

    # Annotated ndarray because that is what every consumer sees after dense_A();
    # the sparse arm is deliberately outside the annotation, exactly as dense_Q's
    # producer is (c525f519).
    return cast(np.ndarray, _sp.csr_matrix((v, (r, c)), shape=(m, n)))


def _append_row_coo(
    rows: list[int], cols: list[int], vals: list[float], r: int, terms: dict[int, float]
) -> None:
    """Append row ``r``'s entries from a ``{column: coefficient}`` mapping."""
    for _j, _v in terms.items():
        rows.append(r)
        cols.append(_j)
        vals.append(_v)


def _append_dense_row_coo(
    rows: list[int], cols: list[int], vals: list[float], r: int, vec: np.ndarray
) -> None:
    """Append only the nonzeros of the dense coefficient row ``vec`` as row ``r``.

    For the extractors whose row source is already a dense vector (the QCP
    walks). Converting each row as it is produced and
    dropping it holds peak memory at O(nnz) rather than O(m*n).

    ``np.nonzero`` retains NaN/inf (they compare unequal to 0), so the finiteness
    guard in ``_extract_lp_data_from_repr`` still sees them.
    """
    (nz,) = np.nonzero(vec)
    if nz.size == 0:
        return
    rows.extend([r] * int(nz.size))
    cols.extend(nz.tolist())
    vals.extend(np.asarray(vec, dtype=np.float64)[nz].tolist())


def _all_finite(arr) -> bool:
    """``np.isfinite(arr).all()`` for a dense array or a scipy sparse matrix.

    Unstored sparse entries are exactly 0.0, which is finite, so checking ``.data``
    is equivalent to checking the densified matrix — without densifying it.
    """
    if _sp_issparse(arr):
        return bool(np.isfinite(arr.data).all())
    a = np.asarray(arr)
    return bool(a.size == 0 or np.isfinite(a).all())


def _extract_qcp_constraints_algebraic(
    model: Model,
    n_orig: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[QuadraticConstraintData, ...]]:
    """Extract linear and quadratic rows without introducing slack variables."""

    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    # COO accumulators rather than lists of dense rows (#863): a row is reduced to
    # its nonzeros as soon as it is produced, so an (m, n_orig) dense stack is never
    # built. The per-QUADRATIC-row Hessian is accumulated sparsely for the same
    # reason and materialised once, dense or CSR per ``_materialise_Q``'s budget —
    # a LINEAR row never materialises one at all.
    ub_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    ub_rhs: list[float] = []
    eq_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    eq_rhs: list[float] = []
    q_rows: list[QuadraticConstraintData] = []

    for con in constraints:
        q_terms, c_vec, const = _extract_quadratic_terms(con.body, model, n_orig)
        rhs = float(con.rhs) - float(const)
        if _quadratic_terms_nonempty(q_terms):
            Q = _materialise_Q(q_terms, n_orig)
            q_rows.append(
                QuadraticConstraintData(
                    # Preserve sparsity: np.asarray() on a sparse matrix silently
                    # yields a 0-d object array rather than raising (#863).
                    # Consumers densify via dense_Q().
                    Q=Q if _sp_issparse(Q) else np.asarray(Q),  # type: ignore[arg-type]
                    c=np.asarray(c_vec),  # type: ignore[arg-type]
                    sense=con.sense,
                    rhs=rhs,
                )
            )
            continue

        if con.sense == "==":
            _append_dense_row_coo(*eq_coo, len(eq_rhs), c_vec)
            eq_rhs.append(rhs)
        elif con.sense == "<=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), c_vec)
            ub_rhs.append(rhs)
        elif con.sense == ">=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), -c_vec)
            ub_rhs.append(-rhs)
        else:
            raise _NotQuadraticError(f"Unknown constraint sense: {con.sense}")

    A_ub = _materialise_A(*ub_coo, len(ub_rhs), n_orig)
    b_ub = np.asarray(ub_rhs, dtype=np.float64)
    A_eq = _materialise_A(*eq_coo, len(eq_rhs), n_orig)
    b_eq = np.asarray(eq_rhs, dtype=np.float64)
    return A_ub, b_ub, A_eq, b_eq, tuple(q_rows)


def extract_lp_data_algebraic(model: Model) -> LPData:
    """Extract LP standard form by walking the expression DAG algebraically.

    Much faster than extract_lp_data() because it avoids JAX tracing/autodiff.
    Returns numpy arrays — the solver converts to jnp at solve time.

    Raises _NotLinearError if the model is not linear.
    """

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    c, obj_const = _extract_linear_coefficients(obj_expr, model, n_orig)

    A_eq, b_eq, x_l, x_u, n_slack = _extract_constraints_algebraic(model, n_orig)
    c_full = np.concatenate([c, np.zeros(n_slack, dtype=np.float64)])

    # Handle objective sense: negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_full = -c_full
        obj_const = -obj_const

    return LPData(
        c=np.asarray(c_full),  # type: ignore[arg-type]
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data_algebraic(model: Model) -> QPData:
    """Extract QP standard form by walking the expression DAG algebraically.

    Much faster than extract_qp_data() because it avoids jax.hessian tracing.
    Returns numpy arrays — the solver converts to jnp at solve time.

    Raises _NotQuadraticError if the objective is not quadratic.
    """

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    Q, c_vec, obj_const = _extract_quadratic_coefficients(obj_expr, model, n_orig)

    A_eq, b_eq, x_l, x_u, n_slack = _extract_constraints_algebraic(model, n_orig)

    if n_slack > 0:
        n_total = n_orig + n_slack
        import scipy.sparse as _sp

        if _sp.issparse(Q):
            # Pad sparsely: densifying here would defeat the whole point (#863).
            Q_full = _sp.block_diag((Q, _sp.csr_matrix((n_slack, n_slack))), format="csr")
        else:
            Q_full = np.zeros((n_total, n_total), dtype=np.float64)
            Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    # Handle objective sense: negate for maximization
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q_full = -Q_full
        c_full = -c_full
        obj_const = -obj_const

    return QPData(
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a 0-d
        # object array rather than raising, which would smuggle garbage into every
        # consumer (#863). Consumers densify via dense_Q().
        Q=Q_full if _sp_issparse(Q_full) else np.asarray(Q_full),  # type: ignore[arg-type]
        c=np.asarray(c_full),  # type: ignore[arg-type]
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qcp_data_algebraic(model: Model) -> QCPData:
    """Extract QCP/QCQP data by walking the expression DAG algebraically."""

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    assert model._objective is not None
    obj_expr = model._objective.expression

    Q, c_vec, obj_const = _extract_quadratic_coefficients(obj_expr, model, n_orig)
    A_ub, b_ub, A_eq, b_eq, q_rows = _extract_qcp_constraints_algebraic(model, n_orig)
    x_l, x_u = _get_variable_bounds(model)

    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q = -Q
        c_vec = -c_vec
        obj_const = -obj_const

    return QCPData(
        # ``Q`` may now be sparse too (#863) — same np.asarray hazard, same fix.
        Q=Q if _sp_issparse(Q) else np.asarray(Q),  # type: ignore[arg-type]
        c=np.asarray(c_vec),  # type: ignore[arg-type]
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a 0-d
        # object array rather than raising, which would smuggle garbage into every
        # consumer (#863). Consumers densify via dense_A().
        A_ub=A_ub if _sp_issparse(A_ub) else np.asarray(A_ub),  # type: ignore[arg-type]
        b_ub=np.asarray(b_ub),  # type: ignore[arg-type]
        A_eq=A_eq if _sp_issparse(A_eq) else np.asarray(A_eq),  # type: ignore[arg-type]
        b_eq=np.asarray(b_eq),  # type: ignore[arg-type]
        quadratic_constraints=q_rows,
        x_l=np.asarray(x_l),  # type: ignore[arg-type]
        x_u=np.asarray(x_u),  # type: ignore[arg-type]
        obj_const=obj_const,
    )


def _linear_terms_from_repr(repr_, n: int, constraint: int | None):
    """``(terms, const)`` for one linear row: ``{col: coef}`` and the value at 0.

    ``constraint is None`` selects the objective; otherwise row ``constraint``.

    Read off the expression arena. ``quadratic_form`` returns a symbolic
    ``(quadratic, linear, constant)`` decomposition; a row is linear exactly when
    its quadratic map is empty, and then its linear map *is* the row -- in time
    proportional to the DAG, not to ``n``. The walk prunes exact zeros, so an
    empty quadratic map is a stronger statement than its degree counter.

    There used to be a numeric probe here, recovering the row by evaluating the
    Rust repr at ``n`` unit vectors (``A_ij = g_i(e_j) - g_i(0)``). A model with
    ``m`` rows cost ``m * n`` full constraint evaluations plus ``m * n`` dense
    ``(n,)`` allocations; on ``acopf_case13659pegase_qcqp`` (n = 199,281,
    m = 191,097) that is 3.8e10 evaluations to recover coefficients the arena was
    already holding. It is gone -- see the module header.

    Raises :class:`_NotQuadraticError` when the walk declines. It does NOT fall
    back here: the only caller is :func:`_extract_lp_data_from_repr`, one rung of
    the ``extract_lp_data`` ladder, and the rungs below it -- the algebraic walk,
    then :func:`_extract_lp_data_tape` -- are the fallback. Catching it here
    would shadow the tape rung, which is the one that handles the vector-valued
    bodies this function cannot represent at all (#75).
    """
    form = (
        repr_.objective_quadratic_form()
        if constraint is None
        else repr_.constraint_quadratic_form(constraint)
    )
    if form is None or len(form[0]) != 0:
        raise _NotQuadraticError(
            f"the arena walk did not prove "
            f"{'the objective' if constraint is None else f'constraint {constraint}'} "
            f"linear (a second-order term, an unsupported construct, a "
            f"vector-valued body, or over the term budget)"
        )
    ci, cd, const = form[3], form[4], form[5]
    return {int(j): float(v) for j, v in zip(ci, cd) if v != 0.0}, float(const)


class _LPRows(NamedTuple):
    """The constraint half of a standard-form extraction, with no objective.

    ``A_eq`` and ``b_eq`` already carry the slack block; ``n_slack`` says how many
    of ``A_eq``'s columns it is, so a caller can pad its own objective to match.
    """

    A_eq: object
    b_eq: np.ndarray
    x_l: np.ndarray
    x_u: np.ndarray
    n_slack: int


def _lp_rows_from_repr(model: Model, repr_) -> _LPRows:
    """Constraints, slacks and bounds off the Rust arena. No objective.

    Split out of :func:`_extract_lp_data_from_repr` because it has two callers
    that want different halves: the LP extractor, which adds a linear objective,
    and :func:`_assemble_qp_from_repr`, which brings a *quadratic* one.

    That second caller is why the split is a correctness fix and not tidying. It
    used to call the whole LP extractor and discard the objective it returned.
    While the objective arm was a numeric probe that silently projected a
    quadratic objective onto its linear part, discarding the answer was merely
    wasteful. The arm reads the arena now and refuses a nonlinear objective
    (:func:`_linear_terms_from_repr`), so routing a QP through it raised on every
    quadratic objective -- every model the QP path exists for.

    Rows are reduced to their nonzeros as they are produced (#863): retaining
    ``n_con`` dense ``(n_orig,)`` vectors and ``np.stack``-ing them is 91.5 GB on
    ``watercontamination0202`` and needs every row resident at once.
    """
    n_orig = repr_.n_vars
    n_con = repr_.n_constraints

    eq_terms: list[dict[int, float]] = []
    eq_rhs: list[float] = []
    ineq_terms: list[dict[int, float]] = []
    ineq_senses: list[str] = []
    ineq_rhs: list[float] = []

    for i in range(n_con):
        sense = repr_.constraint_sense(i)
        rhs_val = repr_.constraint_rhs(i)
        row_terms, g_at_zero = _linear_terms_from_repr(repr_, n_orig, i)

        if sense == "==":
            eq_terms.append(row_terms)
            eq_rhs.append(rhs_val - g_at_zero)
        elif sense == "<=":
            ineq_terms.append(row_terms)
            ineq_senses.append("le")
            ineq_rhs.append(rhs_val - g_at_zero)
        elif sense == ">=":
            ineq_terms.append(row_terms)
            ineq_senses.append("ge")
            ineq_rhs.append(rhs_val - g_at_zero)

    n_eq = len(eq_terms)
    n_ineq = len(ineq_terms)
    n_slack = n_ineq
    n_total = n_orig + n_slack

    coo_rows: list[int] = []
    coo_cols: list[int] = []
    coo_vals: list[float] = []
    b_vals: list[float] = []

    for i in range(n_eq):
        _append_row_coo(coo_rows, coo_cols, coo_vals, i, eq_terms[i])
        b_vals.append(eq_rhs[i])

    for i in range(n_ineq):
        r = n_eq + i
        _append_row_coo(coo_rows, coo_cols, coo_vals, r, ineq_terms[i])
        # body <= 0 becomes body + s = 0; body >= 0 becomes body - s = 0; s >= 0.
        coo_rows.append(r)
        coo_cols.append(n_orig + i)
        coo_vals.append(1.0 if ineq_senses[i] == "le" else -1.0)
        b_vals.append(ineq_rhs[i])

    m_total = n_eq + n_ineq
    A_eq = _materialise_A(coo_rows, coo_cols, coo_vals, m_total, n_total)
    b_eq = np.array(b_vals, dtype=np.float64)

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, np.inf, dtype=np.float64)])

    # This extractor reduces each constraint to a single scalar row. Vector-/
    # matrix-valued constraints (DAE collocation residuals, `Variable @ Constant`
    # MOL stencils) cannot be represented that way, so a non-finite coefficient
    # here means the repr path silently mis-extracted the model. Decline instead
    # of returning corrupt data: `extract_lp_data` then falls through to the tape
    # and autodiff paths, which expand such constraints into one row per
    # component. (A NaN reaching the LP solver otherwise crashes/hangs HiGHS —
    # issue surfaced via test_mol_collocation_solves.)
    for _name, _arr in (("A_eq", A_eq), ("b_eq", b_eq)):
        if not _all_finite(_arr):
            raise _NotLinearError(
                f"repr-based LP extraction produced non-finite {_name}; the model "
                "has vector-valued constraints that are not scalar-representable"
            )

    return _LPRows(A_eq=A_eq, b_eq=b_eq, x_l=x_l, x_u=x_u, n_slack=n_slack)


def _extract_lp_data_from_repr(model: Model) -> LPData:
    """Extract LP standard form by reading the coefficients off the Rust arena.

    One :func:`_linear_terms_from_repr` call per row -- that function does the
    reading, and documents what replaced the unit-vector probe this used to run --
    over the constraint block :func:`_lp_rows_from_repr` builds. Works for
    fast-API and ``from_nl`` models alike, where Python expression trees don't
    exist for the algebraic walk to traverse.
    """
    from discopt._rust import model_to_repr

    repr_ = model_to_repr(model, getattr(model, "_builder", None))
    n_orig = repr_.n_vars

    rows = _lp_rows_from_repr(model, repr_)
    A_eq, b_eq, x_l, x_u, n_slack = rows

    obj_terms, obj_at_zero = _linear_terms_from_repr(repr_, n_orig, None)
    c_full = np.zeros(n_orig + n_slack, dtype=np.float64)
    for _j, _v in obj_terms.items():
        c_full[_j] = _v

    if repr_.objective_sense == "maximize":
        c_full = -c_full
        obj_at_zero = -obj_at_zero

    if not _all_finite(c_full):
        raise _NotLinearError(
            "repr-based LP extraction produced non-finite c; the model has a "
            "vector-valued objective that is not scalar-representable"
        )

    return LPData(
        c=np.asarray(c_full),  # type: ignore[arg-type]
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a 0-d
        # object array rather than raising, which would smuggle garbage into every
        # consumer (#863). Consumers densify via dense_A().
        A_eq=A_eq if _sp_issparse(A_eq) else np.asarray(A_eq),  # type: ignore[arg-type]
        b_eq=np.asarray(b_eq),  # type: ignore[arg-type]
        x_l=np.asarray(x_l),  # type: ignore[arg-type]
        x_u=np.asarray(x_u),  # type: ignore[arg-type]
        obj_const=obj_at_zero,
    )


def _extract_qp_data_symbolic(model: Model) -> QPData:
    """Extract QP standard form by reading the coefficients off the expression DAG.

    This is the first rung of the ``extract_qp_data`` ladder.
    The Rust arena already walks the whole DAG to answer ``is_quadratic`` -- a
    *degree* question whose answer is a bool -- and
    ``ModelRepr.objective_quadratic_form`` emits the coefficients that walk
    already sees, as a sparse COO triplet, in O(nodes).

    Why this exists. The probe recovers the same numbers from
    ``f(e_i + e_j) - f(e_i) - f(e_j) + f(0)``, one model evaluation per variable
    *pair*: O(|support|^2). Measured over the 150-instance MINLPLib MIQP family
    (BQP/IQP/MBQP/MIQP), probing costs **71,330 s** and this walk costs
    **5.99 s** -- and the walk declines on none of them. Per instance the worst
    case is ``unitcommit_200_100_1_mod_8`` (n = 25,700): 28,288 s of probing
    against 0.042 s here, for a Q with 4,662 nonzeros whose dense form is 5.3 GB.

    It is also *more accurate*, not merely faster. Every coefficient here is a
    sum of products of literals in the DAG, so there is no subtractive
    cancellation; the probe identities are differences of nearly-equal floats,
    which is what returned ``Q = 0`` for a sum of squares and certified a false
    optimum in #866. On ``chimera_mis-01`` the probe spends 275 s and then fails
    its own #866 verification (-171.42 against a true -174.09), so the work is
    discarded and redone by the tape.

    Verified against ``repr_.evaluate_objective`` at 5 random points on each of
    the 150 instances: 750 comparisons, worst relative error 6.9e-14, zero
    mismatches.

    Raises :class:`_NotQuadraticError` when the walk declines -- the objective
    is not a quadratic form, it uses a construct the walk does not represent, or
    its coefficients would not fit the term budget. The walk never approximates,
    so a decline is the only failure mode and the dispatcher falls through.
    """
    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)
    n_orig = repr_.n_vars

    form = repr_.objective_quadratic_form()
    if form is None:
        raise _NotQuadraticError(
            "the objective is not representable as a quadratic form by the "
            "symbolic DAG walk (not degree 2, an unsupported construct, or over "
            "the coefficient budget) — falling through to the next extractor"
        )
    Q, c_vec, obj_const = _quadratic_form_to_coefficients(form, n_orig)
    return _assemble_qp_from_repr(model, repr_, n_orig, Q, c_vec, obj_const)


def _quadratic_form_to_coefficients(form, n: int):
    """Convert a ``(qi, qj, qd, ci, cd, constant)`` COO quadratic form -- what
    ``ModelRepr.objective_quadratic_form`` / ``.constraint_quadratic_form``
    return -- into the ``0.5 x' Q x + c'x + d`` triple every consumer here
    expects.

    The walk returns the FULL coefficient of ``x_i * x_j``; QPData carries the
    ``0.5 x' Q x`` convention with a symmetric Q. So a diagonal coefficient
    doubles (``0.5 * Q[i,i] * x_i^2 == qd * x_i^2`` requires ``Q[i,i] = 2*qd``)
    and an off-diagonal splits across the two symmetric halves
    (``0.5 * 2 * Q[i,j] == qd`` requires ``Q[i,j] = Q[j,i] = qd``).

    The doubling is on THIS arm only. :func:`_tape_quadratic_coefficients`, the
    fallback when the walk declines, returns a Hessian -- and the Hessian of
    ``0.5 x' Q x`` is already ``Q``, diagonal included -- so applying the same
    correction there would double every diagonal a second time. Each function
    documents its own convention because the two arms feed the same consumers.

    Shared by the objective arm and the quadratic-constraint arm so the two
    conventions cannot drift apart.
    """
    qi, qj, qd, ci, cd, const = form
    qi = np.asarray(qi, dtype=np.int64)
    qj = np.asarray(qj, dtype=np.int64)
    qd = np.asarray(qd, dtype=np.float64)

    _is_diag = qi == qj
    rows = np.concatenate([qi[_is_diag], qi[~_is_diag], qj[~_is_diag]])
    cols = np.concatenate([qj[_is_diag], qj[~_is_diag], qi[~_is_diag]])
    vals = np.concatenate([2.0 * qd[_is_diag], qd[~_is_diag], qd[~_is_diag]])

    c_vec = np.zeros(n, dtype=np.float64)
    if len(ci):
        c_vec[np.asarray(ci, dtype=np.int64)] = np.asarray(cd, dtype=np.float64)

    # Dense while it comfortably fits, sparse beyond that (#863) -- consumers
    # accept either, and a dense Q is 5.3 GB at n = 25,700.
    if (n * n * 8) <= _QP_DENSE_Q_MAX_BYTES:
        Q = np.zeros((n, n), dtype=np.float64)
        if len(rows):
            Q[rows, cols] = vals
    else:
        import scipy.sparse as _sp

        Q = cast(np.ndarray, _sp.csr_matrix((vals, (rows, cols)), shape=(n, n)))

    return Q, c_vec, float(const)


def _quadratic_coefficients(
    repr_, n_vars: int, constraint: int | None, tape_factory=None, maximize: bool = False
):
    """``(Q, c, d)`` for the objective (``constraint is None``) or one constraint.

    Reads the coefficients off the expression arena. When the walk declines -- an
    unsupported construct, a vector-valued body, or a form over the term budget --
    falls back to :func:`_tape_quadratic_coefficients`, one AD-tape Hessian for
    that row. Both arms are exact; neither probes.

    ``tape_factory`` is a zero-argument callable returning the shared
    :class:`TapeNLPEvaluator` for this model (or ``None`` when the model is not
    representable). A factory rather than an evaluator for two reasons: it is
    called only when the walk actually declines, so the common path never pays to
    build a tape; and the caller memoises it, so a model with ``m`` declining rows
    builds one tape rather than ``m``. ``tape_factory=None`` means the caller has
    no tape to offer, and a decline is then a loud :class:`_NotQuadraticError`
    rather than a guess.

    This is the QCP/QCQP counterpart of :func:`_extract_qp_data_symbolic`, and the
    per-row cost mattered more here than on the objective: the numeric probe this
    replaced ran **once per constraint**, so a model with ``m`` quadratic rows paid
    ``m`` separate O(|support|^2) pair sweeps.
    """
    form = (
        repr_.objective_quadratic_form()
        if constraint is None
        else repr_.constraint_quadratic_form(constraint)
    )
    if form is not None:
        return _quadratic_form_to_coefficients(form, n_vars)

    tape = tape_factory() if tape_factory is not None else None
    if tape is None:
        raise _NotQuadraticError(
            f"the arena walk declined "
            f"{'the objective' if constraint is None else f'constraint {constraint}'} "
            f"(unsupported construct, vector-valued body, or over the term budget) "
            f"and no AD tape is available to fall back on"
        )
    return _tape_quadratic_coefficients(tape, n_vars, constraint, maximize=maximize)


def _tape_quadratic_coefficients(tape, n_vars: int, constraint: int | None, maximize: bool = False):
    """``(Q, c, d)`` for one row from the AD tape -- the analytical fallback.

    The row is quadratic by classification, so its Hessian is constant and one
    evaluation at the origin gives it exactly.

    In the ``0.5 x' Q x + c'x + d`` convention every consumer here uses, ``Q`` IS
    the Hessian (the Hessian of ``0.5 x' Q x`` is ``Q``), so no factor of two is
    applied on this arm. :func:`_quadratic_form_to_coefficients` documents why the
    arena arm needs one and this arm does not -- the walk emits the FULL
    coefficient of ``x_i x_j``, a second derivative is already doubled on the
    diagonal.

    ``evaluate_lagrangian_hessian(x, obj_factor=0, lambda_=e_i)`` isolates
    ``H(g_i)``: it is the only way to get a single constraint's curvature out of a
    Lagrangian evaluator, and a nonzero ``obj_factor`` here would silently add the
    objective's Hessian into every constraint row.

    ``maximize`` is the MODEL's sense, passed in rather than read off the tape --
    the tape's own flag is private and means "I already negated it", which is the
    opposite question. The objective arm undoes that flip so it returns in the
    model's own sense, matching the arena arm, because the caller applies one
    shared negation to whichever arm ran. :func:`_qp_terms_tape` documents the
    same correction, where omitting it handed a maximisation to a minimiser and
    produced a false optimum. Constraint rows are never flipped.
    """
    x0 = np.zeros(n_vars, dtype=np.float64)
    if constraint is None:
        Q = np.asarray(tape.evaluate_hessian(x0), dtype=np.float64)
        c_vec = np.asarray(tape.evaluate_gradient(x0), dtype=np.float64)
        const = float(tape.evaluate_objective(x0))
        if maximize:
            Q, c_vec, const = -Q, -c_vec, -const
        return Q, c_vec, const

    lam = np.zeros(tape.n_constraints, dtype=np.float64)
    lam[constraint] = 1.0
    Q = np.asarray(tape.evaluate_lagrangian_hessian(x0, 0.0, lam), dtype=np.float64)
    c_vec = np.asarray(tape.evaluate_jacobian(x0), dtype=np.float64)[constraint]
    const = float(np.asarray(tape.evaluate_constraints(x0), dtype=np.float64)[constraint])
    return Q, np.asarray(c_vec, dtype=np.float64), const


def _extraction_tape(model: Model, repr_, n_orig: int):
    """A tape evaluator aligned to ``repr_``, or ``None`` when there isn't one.

    Both guards are load-bearing. ``Q``/``c`` are indexed by this module's flat
    variable order and rows by ``repr_``'s constraint order; a tape disagreeing
    about either would be silently MISALIGNED rather than wrong-loud -- which is
    the failure mode :func:`_qp_terms_tape` already guards the variable half of.
    """
    from discopt._tape_nlp_evaluator import try_build

    ev = try_build(model)
    if ev is None:
        return None
    if ev.n_variables != n_orig or ev.n_constraints != repr_.n_constraints:
        logger.debug(
            "extraction tape rejected: tape is (%d vars, %d cons), repr is (%d, %d)",
            ev.n_variables,
            ev.n_constraints,
            n_orig,
            repr_.n_constraints,
        )
        return None
    return ev


def _assemble_qp_from_repr(model, repr_, n_orig: int, Q, c_vec, d: float) -> QPData:
    """Attach constraints, slack padding and objective sense to a recovered
    ``(Q, c, d)`` and package it as :class:`QPData`.

    Shared by every rung that recovers ``(Q, c, d)`` from the repr, so they
    cannot drift: they differ only in how ``Q`` and ``c`` are recovered, and
    everything downstream of that -- the LP constraint extraction, the
    slack block, the maximize negation (#28) -- is this one function.

    ``Q`` follows the ``0.5 x' Q x + c'x + d`` convention and is symmetric;
    it may be dense or sparse and is passed through unchanged.
    """
    # Constraints only: this caller supplies its own (quadratic) objective, and
    # the arena walk would refuse to read that one as linear. See _lp_rows_from_repr.
    rows = _lp_rows_from_repr(model, repr_)
    n_slack = rows.n_slack

    if n_slack > 0:
        n_total = n_orig + n_slack
        import scipy.sparse as _sp

        if _sp.issparse(Q):
            # Pad sparsely: densifying here would defeat the whole point (#863).
            Q_full = _sp.block_diag((Q, _sp.csr_matrix((n_slack, n_slack))), format="csr")
        else:
            Q_full = np.zeros((n_total, n_total), dtype=np.float64)
            Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    # Maximize → minimize -f: the QP backends always minimize, so negate the
    # whole quadratic (Q, c, constant). Without this the repr path returns the
    # raw maximize form — an indefinite Q for a concave-maximize objective — which
    # the QP solver rejects (and the autodiff fallback would have handled
    # correctly), silently yielding a wrong optimum. Mirrors the negation in
    # `_extract_lp_data_from_repr` and `_extract_qp_data_autodiff`. (Surfaced via
    # test_maximize_objective_sign_not_negated, issue #28.)
    if repr_.objective_sense == "maximize":
        Q_full = -Q_full
        c_full = -c_full
        d = -d

    return QPData(
        # Preserve sparsity: np.asarray() on a sparse matrix silently yields a
        # 0-d object array rather than raising, which would smuggle garbage into
        # every consumer (#863). Consumers densify via dense_Q().
        Q=Q_full if _sp_issparse(Q_full) else np.asarray(Q_full),  # type: ignore[arg-type]
        c=np.asarray(c_full),  # type: ignore[arg-type]
        A_eq=rows.A_eq,  # type: ignore[arg-type]
        b_eq=rows.b_eq,  # type: ignore[arg-type]
        x_l=rows.x_l,  # type: ignore[arg-type]
        x_u=rows.x_u,  # type: ignore[arg-type]
        obj_const=d,
    )


def _extract_qcp_data_from_repr(model: Model) -> QCPData:
    """Extract QCP/QCQP data by evaluating the Rust ModelRepr."""

    from discopt._rust import model_to_repr

    _builder = getattr(model, "_builder", None)
    repr_ = model_to_repr(model, _builder)

    n_orig = repr_.n_vars

    # One-slot memo. An empty list means "not built yet"; a one-element list means
    # "built", and that one element may legitimately be None (the model is not
    # tape-representable). Distinguishing those two states is the whole point -- a
    # plain ``None`` sentinel would rebuild the tape on every declining row.
    _tape_memo: list = []

    def _tape():
        if not _tape_memo:
            _tape_memo.append(_extraction_tape(model, repr_, n_orig))
        return _tape_memo[0]

    Q, c_vec, obj_const = _quadratic_coefficients(
        repr_, n_orig, None, tape_factory=_tape, maximize=repr_.objective_sense == "maximize"
    )

    # COO accumulators rather than lists of dense rows (#863); see _materialise_A.
    ub_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    ub_rhs: list[float] = []
    eq_coo: tuple[list[int], list[int], list[float]] = ([], [], [])
    eq_rhs: list[float] = []
    q_rows: list[QuadraticConstraintData] = []

    for i in range(repr_.n_constraints):
        row_Q, row_c, row_const = _quadratic_coefficients(repr_, n_orig, i, tape_factory=_tape)
        sense = repr_.constraint_sense(i)
        rhs = float(repr_.constraint_rhs(i)) - float(row_const)
        if _quadratic_row_has_terms(row_Q):
            q_rows.append(
                QuadraticConstraintData(
                    # Preserve sparsity (see dense_Q / #863): np.asarray() on a scipy
                    # sparse matrix returns a 0-d object array instead of raising.
                    Q=row_Q if _sp_issparse(row_Q) else np.asarray(row_Q),  # type: ignore[arg-type]
                    c=np.asarray(row_c),  # type: ignore[arg-type]
                    sense=sense,
                    rhs=rhs,
                )
            )
            continue
        if sense == "==":
            _append_dense_row_coo(*eq_coo, len(eq_rhs), row_c)
            eq_rhs.append(rhs)
        elif sense == "<=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), row_c)
            ub_rhs.append(rhs)
        elif sense == ">=":
            _append_dense_row_coo(*ub_coo, len(ub_rhs), -row_c)
            ub_rhs.append(-rhs)

    x_l, x_u = _get_variable_bounds(model)
    A_ub = _materialise_A(*ub_coo, len(ub_rhs), n_orig)
    b_ub = np.asarray(ub_rhs, dtype=np.float64)
    A_eq = _materialise_A(*eq_coo, len(eq_rhs), n_orig)
    b_eq = np.asarray(eq_rhs, dtype=np.float64)

    if repr_.objective_sense == "maximize":
        Q = -Q
        c_vec = -c_vec
        obj_const = -obj_const

    return QCPData(
        # Preserve sparsity (see dense_Q / #863, #875).
        Q=Q if _sp_issparse(Q) else np.asarray(Q),  # type: ignore[arg-type]
        c=np.asarray(c_vec),  # type: ignore[arg-type]
        # Preserve sparsity (see dense_A / #863).
        A_ub=A_ub if _sp_issparse(A_ub) else np.asarray(A_ub),  # type: ignore[arg-type]
        b_ub=np.asarray(b_ub),  # type: ignore[arg-type]
        A_eq=A_eq if _sp_issparse(A_eq) else np.asarray(A_eq),  # type: ignore[arg-type]
        b_eq=np.asarray(b_eq),  # type: ignore[arg-type]
        quadratic_constraints=tuple(q_rows),
        x_l=np.asarray(x_l),  # type: ignore[arg-type]
        x_u=np.asarray(x_u),  # type: ignore[arg-type]
        obj_const=float(obj_const),
    )


def extract_lp_data(model: Model) -> LPData:
    """Extract LP standard form from a model classified as LP.

    Tries Rust repr-based extraction first (for fast-API models), then
    algebraic extraction (for expression-based), then falls back to
    autodiff-based extraction if the DAG walk fails.

    Inequality constraints are converted to equalities with slacks:
      - body <= 0 becomes body + s = 0, s >= 0
      - body >= 0 becomes body - s = 0, s >= 0

    Args:
        model: A Model classified as ProblemClass.LP.

    Returns:
        LPData with c, A_eq, b_eq, x_l, x_u.
    """
    # The arena walk goes first, unconditionally. It used to be gated on
    # ``_builder is not None`` with a second, ungated copy of the same call further
    # down the ladder -- because it was a numeric probe then, cheap on a fast-API
    # model and something you wanted to defer on anything else. It reads the
    # arena now (see ``_linear_terms_from_repr``), so it is O(nodes) either way
    # and there is exactly one rung. The ``_builder`` gate was never about
    # correctness: ``model_to_repr`` accepts ``_builder=None``.
    try:
        return _extract_lp_data_from_repr(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the algebraic extractor
        # Each rung of this ladder is a *fast path*: a silent fall-through turns
        # "the repr extractor declined" into an unexplained measurement.
        logger.debug("LP repr extraction declined: %s: %s", type(exc).__name__, exc)

    try:
        return extract_lp_data_algebraic(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the tape/autodiff extractors
        logger.debug("LP algebraic extraction declined: %s: %s", type(exc).__name__, exc)

    # #75: the last JAX-free rung. The three above all reduce a constraint to one
    # scalar row, so a vector-valued body (DAE collocation residual, MOL spatial
    # stencil) reached `_extract_lp_data_autodiff` and imported JAX on a default
    # solve. The tape fans such a body out into one row per component, which is
    # the only thing `jax.jacobian` was still being used for here.
    _tape_lp = _extract_lp_data_tape(model)
    if _tape_lp is not None:
        return _tape_lp

    return _extract_lp_data_autodiff(model)


def _extract_lp_data_tape(model: Model) -> LPData | None:
    """Extract LP standard form from the JAX-free tape evaluator, or ``None``.

    This is the rung that takes **vector-valued constraint bodies** off JAX.
    Every other JAX-free extractor collapses a constraint to one scalar row:
    ``_extract_lp_data_from_repr`` reads one scalar row per constraint off the
    arena, and cannot represent an array-valued body at all, while
    ``_extract_linear_coefficients_sparse`` refuses an array variable in scalar
    position outright (C-29 — collapsing it to one summed row certifies
    infeasible points). So a DAE collocation model or an MOL spatial stencil
    fell all the way through to ``_extract_lp_data_autodiff`` and imported JAX on
    a default solve; measured on a 3-element/3-point Radau collocation model,
    210 ``jax*`` modules.

    ``TapeNLPEvaluator`` already has exactly the fan-out ``jax.jacobian`` was
    being used for: ``_build`` compiles each body with
    ``compile_to_nl_array(...).reshape(-1)``, so an array-valued body is one
    ``Constraint`` and many tape rows, in the same C order the JAX evaluator
    concatenates. ``constraint_row_map()`` gives the ``(start, stop,
    Constraint)`` back, which is what turns tape rows into LP rows with the
    right sense and the right slack column.

    Returns ``None`` — never a guess — when the tape cannot represent the model,
    when its variable layout disagrees with this module's flat order, or when
    the affine check below fails; the caller then falls back to JAX.
    """
    from discopt._tape_nlp_evaluator import try_build
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    ev = try_build(model)
    if ev is None:
        return None
    if ev.n_variables != n_orig:
        # Same guard as `_qp_terms_tape`: the columns are this module's flat slot
        # order, so a differing width means the two disagree about the layout and
        # every coefficient would land in the wrong column — silently.
        logger.debug(
            "LP tape extraction skipped: tape has %d variables, model flat width is %d",
            ev.n_variables,
            n_orig,
        )
        return None

    x_zero = np.zeros(n_orig, dtype=np.float64)
    body0 = np.asarray(ev.evaluate_constraints(x_zero), dtype=np.float64).reshape(-1)
    m_rows = int(body0.shape[0])

    # Row -> sense, via the map the tape builds from the same list its rows come
    # from. `""` marks a row no source constraint claimed; that would mean the
    # map and the row list have drifted, so decline rather than emit a row whose
    # sense is a guess.
    row_sense: list[str] = [""] * m_rows
    for start, stop, con in ev.constraint_row_map():
        if con.sense not in ("==", "<=", ">="):
            logger.debug("LP tape extraction declined: unsupported sense %r", con.sense)
            return None
        if stop > m_rows:
            logger.debug("LP tape extraction declined: row map overruns %d tape rows", m_rows)
            return None
        for r in range(start, stop):
            row_sense[r] = con.sense
    if any(s == "" for s in row_sense):
        logger.debug("LP tape extraction declined: row map does not cover every tape row")
        return None

    jac_rows, jac_cols = ev.jacobian_structure()
    jac_vals = np.asarray(ev.evaluate_jacobian_values(x_zero), dtype=np.float64)

    # Soundness gate, not a sanity check. `c`/`A` here are the *affine model* of
    # the objective and bodies at the origin, which is the true model only when
    # both are affine. `classify_problem` says LP/QP, but this extractor is the
    # last rung before JAX and a mis-classification would linearise a nonlinear
    # body and hand the B&B a relaxation that cuts off feasible points — a false
    # `optimal`, with no symptom. Two extra tape evaluations rule that out.
    # Two points, not one: a single point can agree by coincidence (``x*(x -
    # 1/7)`` matches its origin tangent exactly at ``x = 1/7``, which is the
    # first pattern), and the second pattern is not a multiple of the first.
    # Deterministic, not random, so a decline is reproducible.
    lb_p, ub_p = _get_variable_bounds(model)
    for _period, _scale in ((7, 1.0), (5, -0.5)):
        x_probe = np.array(
            [_scale * ((j % _period) + 1) / _period for j in range(n_orig)], dtype=np.float64
        )
        x_probe = np.clip(x_probe, lb_p, ub_p)
        x_probe = np.where(np.isfinite(x_probe), x_probe, 0.0)
        body_probe = np.asarray(ev.evaluate_constraints(x_probe), dtype=np.float64).reshape(-1)
        affine_probe = body0.copy()
        if jac_vals.size:
            np.add.at(affine_probe, jac_rows, jac_vals * x_probe[jac_cols])
        tol = 1e-7 * (1.0 + np.abs(body_probe))
        if body_probe.shape != affine_probe.shape or not np.all(
            np.abs(body_probe - affine_probe) <= tol
        ):
            logger.debug("LP tape extraction declined: constraint bodies are not affine")
            return None

    # `c`/`obj_const` are the affine model of the objective at the origin, which
    # is what `jax.grad(obj_fn)(0)` / `obj_fn(0)` give in the JAX arm. No affine
    # gate on the objective, deliberately: `_extract_qp_data_autodiff` calls this
    # function *for the constraints* on a QP, where the objective is quadratic by
    # construction and this `c` is discarded. Gating on it would decline every QP.
    c_vec = np.asarray(ev.evaluate_gradient(x_zero), dtype=np.float64)
    obj_const = float(ev.evaluate_objective(x_zero))

    assert model._objective is not None
    _maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
    if _maximize:
        # Back to the MODEL's sense: `TapeNLPEvaluator._build` negates a MAXIMIZE
        # objective, so the tape already minimises. Undo it here and let the
        # shared negation at the tail apply, so this arm's sense handling is
        # literally the same code as every other extractor's. Eliding this pair
        # is what made `_qp_terms_tape` hand a maximisation to a minimiser.
        c_vec = -c_vec
        obj_const = -obj_const

    # Equalities first, then inequalities with one slack each — the row order
    # every other extractor in this module produces.
    eq_src = [r for r in range(m_rows) if row_sense[r] == "=="]
    ineq_src = [r for r in range(m_rows) if row_sense[r] != "=="]
    n_eq = len(eq_src)
    n_slack = len(ineq_src)
    n_total = n_orig + n_slack

    new_row = np.full(m_rows, -1, dtype=np.int64)
    for k, r in enumerate(eq_src):
        new_row[r] = k
    for k, r in enumerate(ineq_src):
        new_row[r] = n_eq + k

    if m_rows and n_orig and jac_vals.size:
        # Sum duplicate (row, col) entries the way `evaluate_jacobian`'s
        # `np.add.at` does — `_materialise_A` *assigns*, so a repeated structure
        # entry would otherwise drop every contribution but the last.
        key = new_row[jac_rows] * np.int64(n_orig) + jac_cols.astype(np.int64)
        uniq, inverse = np.unique(key, return_inverse=True)
        summed = np.bincount(inverse, weights=jac_vals, minlength=uniq.shape[0])
        coo_rows = (uniq // np.int64(n_orig)).astype(np.intp)
        coo_cols = (uniq % np.int64(n_orig)).astype(np.intp)
        coo_vals = summed.astype(np.float64)
    else:
        coo_rows = np.zeros(0, dtype=np.intp)
        coo_cols = np.zeros(0, dtype=np.intp)
        coo_vals = np.zeros(0, dtype=np.float64)

    # body <= 0 becomes body + s = 0; body >= 0 becomes body - s = 0; s >= 0.
    slack_rows = np.arange(n_eq, n_eq + n_slack, dtype=np.intp)
    slack_cols = np.arange(n_orig, n_orig + n_slack, dtype=np.intp)
    slack_vals = np.array(
        [1.0 if row_sense[r] == "<=" else -1.0 for r in ineq_src], dtype=np.float64
    )

    coo_rows = np.concatenate([coo_rows, slack_rows])
    coo_cols = np.concatenate([coo_cols, slack_cols])
    coo_vals = np.concatenate([coo_vals, slack_vals])

    b_eq = np.zeros(m_rows, dtype=np.float64)
    if m_rows:
        b_eq[new_row] = -body0

    A_eq = _materialise_A(
        coo_rows,  # type: ignore[arg-type]  # arrays, not lists: nnz can be millions
        coo_cols,  # type: ignore[arg-type]
        coo_vals,  # type: ignore[arg-type]
        m_rows,
        n_total,
    )

    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    x_l = np.concatenate([x_l_orig, np.zeros(n_slack, dtype=np.float64)])
    x_u = np.concatenate([x_u_orig, np.full(n_slack, np.inf, dtype=np.float64)])

    # Handle objective sense: negate for maximization (solvers always minimize).
    if _maximize:
        c_full = -c_full
        obj_const = -obj_const

    return LPData(
        # numpy, not jnp — see `_extract_qp_data_autodiff`'s note.
        c=c_full,  # type: ignore[arg-type]
        A_eq=A_eq,  # type: ignore[arg-type]
        b_eq=b_eq,  # type: ignore[arg-type]
        x_l=x_l,  # type: ignore[arg-type]
        x_u=x_u,  # type: ignore[arg-type]
        obj_const=obj_const,
    )


def _extract_lp_data_autodiff(model: Model) -> LPData:
    """Extract LP standard form using autodiff (original slow path).

    Uses ``jax.jacobian`` rather than ``jax.grad`` so that vector-valued
    constraint bodies (DAE collocation residuals, MOL spatial residuals)
    are handled the same way as scalar bodies: each component contributes
    one row in the LP matrix, and inequalities get one slack per row.
    """
    import jax
    import jax.numpy as jnp

    from discopt._relax.dag_compiler import compile_constraint, compile_objective
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    obj_fn = compile_objective(model)

    # Extract c and constant: obj(x) = c'x + d, so grad(obj)(0) = c, obj(0) = d
    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)
    c = jax.grad(obj_fn)(x_zero)
    obj_const = float(obj_fn(x_zero))

    # Extract constraint coefficients
    constraints = [con for con in model._constraints if isinstance(con, Constraint)]

    # First pass: compile each constraint and probe its row count by
    # evaluating at zero. Scalar bodies become a single row; vector bodies
    # contribute one row per component.
    eq_blocks: list[tuple[jnp.ndarray, jnp.ndarray]] = []  # (J, body0)
    ineq_blocks: list[tuple[jnp.ndarray, jnp.ndarray, str]] = []  # (J, body0, sense)

    for con in constraints:
        con_fn = compile_constraint(con, model)
        # Flatten any vector / matrix constraint body into a length-k vector;
        # each component becomes its own LP row.
        body0 = jnp.asarray(con_fn(x_zero), dtype=jnp.float64).reshape(-1)
        jac_raw = jax.jacobian(lambda x, _f=con_fn: jnp.asarray(_f(x)).reshape(-1))(x_zero)
        jac = jnp.asarray(jac_raw, dtype=jnp.float64).reshape(body0.shape[0], n_orig)
        if con.sense == "==":
            eq_blocks.append((jac, body0))
        elif con.sense == "<=":
            ineq_blocks.append((jac, body0, "le"))
        elif con.sense == ">=":
            ineq_blocks.append((jac, body0, "ge"))

    n_eq_rows = sum(int(j.shape[0]) for j, _ in eq_blocks)
    n_ineq_rows = sum(int(j.shape[0]) for j, _, _ in ineq_blocks)
    n_slack = n_ineq_rows
    n_total = n_orig + n_slack

    A_rows: list[jnp.ndarray] = []
    b_vals: list[float] = []

    for jac, body0 in eq_blocks:
        for r in range(jac.shape[0]):
            A_rows.append(jnp.concatenate([jac[r], jnp.zeros(n_slack)]))
            b_vals.append(-float(body0[r]))

    slack_offset = 0
    for jac, body0, sense in ineq_blocks:
        for r in range(jac.shape[0]):
            slack_col = jnp.zeros(n_slack)
            sign = 1.0 if sense == "le" else -1.0
            # body ≤ 0 → body + s = 0, s ≥ 0; body ≥ 0 → body − s = 0, s ≥ 0.
            slack_col = slack_col.at[slack_offset].set(sign)
            A_rows.append(jnp.concatenate([jac[r], slack_col]))
            b_vals.append(-float(body0[r]))
            slack_offset += 1

    m_total = n_eq_rows + n_ineq_rows
    if m_total > 0:
        A_eq = jnp.stack(A_rows)
        b_eq = jnp.array(b_vals, dtype=jnp.float64)
    else:
        A_eq = jnp.zeros((0, n_total), dtype=jnp.float64)
        b_eq = jnp.zeros(0, dtype=jnp.float64)

    # Bounds: original vars keep their bounds, slack vars >= 0
    x_l_orig, x_u_orig = _get_variable_bounds(model)
    c_full = jnp.concatenate([c, jnp.zeros(n_slack)])
    x_l = jnp.concatenate([x_l_orig, jnp.zeros(n_slack)])
    x_u = jnp.concatenate([x_u_orig, jnp.full(n_slack, jnp.inf)])

    # Handle objective sense: negate for maximization (solvers always minimize).
    # C-30: this autodiff fallback previously dropped the maximize negation that
    # every other extractor applies (extract_lp_data_algebraic:743,
    # _extract_lp_data_from_repr:927, _extract_qp_data_autodiff:1375), so a
    # `maximize` model routed here (e.g. a vector `sum(const*var)` body that the
    # algebraic walk refuses) was silently minimized and returned 0.
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        c_full = -c_full
        obj_const = -obj_const

    return LPData(
        c=c_full,
        A_eq=A_eq,
        b_eq=b_eq,
        x_l=x_l,
        x_u=x_u,
        obj_const=obj_const,
    )


def extract_qp_data(model: Model) -> QPData:
    """Extract QP standard form from a model classified as QP.

    Three analytical rungs, in increasing generality and cost: the symbolic walk
    over the Rust expression arena, then the Python-side algebraic walk, then the
    AD tape. Every rung reads the coefficients the model already carries; none of
    them measures the objective. Each declines loudly rather than approximating,
    so falling through costs only the walk that declined.

    There used to be a fourth rung -- a numeric probe -- ahead of the tape, and it
    ran by default. It is gone; see the module header for the measurement and for
    why its position was justified by a comment that had stopped being true.

    Args:
        model: A Model classified as ProblemClass.QP.

    Returns:
        QPData with Q, c, A_eq, b_eq, x_l, x_u.
    """
    # The symbolic walk needs neither a ``_builder`` nor Python-level expression
    # objects, so it covers both the API-built and the ``from_nl`` arms.
    try:
        return _extract_qp_data_symbolic(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the algebraic walk
        # Logged, never silent: a fast path that disappears without evidence that
        # it did is how the probe stayed the default for as long as it did.
        logger.debug("QP symbolic extraction declined: %s: %s", type(exc).__name__, exc)

    try:
        return extract_qp_data_algebraic(model)
    except Exception as exc:  # noqa: BLE001 - falls through to the autodiff extractor
        logger.debug("QP algebraic extraction declined: %s: %s", type(exc).__name__, exc)

    return _extract_qp_data_autodiff(model)


def extract_qcp_data(model: Model) -> QCPData:
    """Extract QCP/QCQP data from a model classified as QCP/QCQP/MIQCP/MIQCQP."""
    _builder = getattr(model, "_builder", None)
    if _builder is not None:
        try:
            return _extract_qcp_data_from_repr(model)
        except Exception as exc:  # noqa: BLE001 - falls through to the algebraic extractor
            logger.debug("QCP repr extraction failed: %s: %s", type(exc).__name__, exc)

    return extract_qcp_data_algebraic(model)


def _qp_terms_tape(model: Model, n_orig: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    """``(Q, c, d)`` of a quadratic objective from the JAX-free tape evaluator.

    Returns ``None`` when the tape cannot represent the model, so the caller
    falls back to the JAX extractor rather than guessing.

    The objective here is quadratic by construction — this function is only
    reached for models ``classify_problem`` put in the QP/MIQP family — so its
    Hessian is constant and one evaluation at the origin gives ``Q`` exactly.
    That is the same identity the JAX branch below uses; only the differentiator
    changes.

    Returned in the MODEL's own sense, exactly as ``_qp_terms_jax`` is, so the
    caller's shared ``MAXIMIZE`` negation applies to both arms identically. The
    tape itself minimises — ``TapeNLPEvaluator._build`` does ``obj = -obj`` for a
    maximisation — so this function has to undo that. Without the undo the caller
    negated an already-negated objective and handed a *maximisation* to a
    minimiser: measured on ``maximize -(x-2)**2`` the tape arm produced
    ``Q=-2, c=+4, d=-4`` where the JAX arm produced ``Q=+2, c=-4, d=+4``. That is
    a false optimum, not a slow path.
    """
    from discopt._tape_nlp_evaluator import try_build
    from discopt.modeling.core import ObjectiveSense

    ev = try_build(model)
    if ev is None:
        return None
    # Guard the flat-slot correspondence: Q/c are indexed by this module's
    # variable order, so a differing width means the two disagree about the
    # layout and the result would be silently misaligned rather than wrong-loud.
    if ev.n_variables != n_orig:
        logger.debug(
            "QP tape extraction skipped: tape has %d variables, model flat width is %d",
            ev.n_variables,
            n_orig,
        )
        return None

    x_zero = np.zeros(n_orig, dtype=np.float64)
    Q = np.asarray(ev.evaluate_hessian(x_zero), dtype=np.float64)
    c_vec = np.asarray(ev.evaluate_gradient(x_zero), dtype=np.float64)
    obj_const = float(ev.evaluate_objective(x_zero))
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        # Undo the tape's internal minimisation flip; see the docstring.
        Q, c_vec, obj_const = -Q, -c_vec, -obj_const
    return Q, c_vec, obj_const


def _qp_terms_jax(model: Model, n_orig: int) -> tuple[np.ndarray, np.ndarray, float]:
    """``(Q, c, d)`` via JAX — the legacy differentiator, kept as the fallback."""
    import jax
    import jax.numpy as jnp

    from discopt._relax.dag_compiler import compile_objective

    obj_fn = compile_objective(model)
    x_zero = jnp.zeros(n_orig, dtype=jnp.float64)

    # Q = hessian(obj) — constant for QP
    Q = np.asarray(jax.hessian(obj_fn)(x_zero), dtype=np.float64)

    # c = grad(obj)(0) = Q*0 + c = c (linear part)
    c_vec = np.asarray(jax.grad(obj_fn)(x_zero), dtype=np.float64)

    # Constant term: f(0) = 0.5*0'Q*0 + c'*0 + d = d
    obj_const = float(obj_fn(x_zero))
    return Q, c_vec, obj_const


def _extract_qp_data_autodiff(model: Model) -> QPData:
    """Extract QP standard form using autodiff (original slow path).

    Tape first, JAX only if the tape cannot represent the model. Until #75 this
    function imported ``jax`` unconditionally, which made it a *default* JAX
    consumer on a branch meant to have none: back then ``extract_qp_data`` fell
    through to here whenever the numeric probe that used to sit above it could not
    reproduce the objective, and that refusal was routine on larger instances
    (measured on ``chimera_mis-01``, a 2032-variable MIQP with no nonlinear
    constraint anywhere: the probe recovered -171.42 against a true -174.09, fell
    through, and the solve imported 210 jax modules via ``qubo_local_search`` →
    ``extract_qp_data``). The probe is deleted and the symbolic walk above answers
    that class outright, so this rung is reached far less often -- but it is still
    the general one, and still the fallback for anything the walks decline.

    Entry experiment before the swap (CLAUDE.md §4), tape phase asserted
    jax-free throughout: on 7 QP/MIQP instances (n = 2 … 2032, incl.
    ``chimera_mis-01``) the tape reproduced the JAX ``Q`` to relative error
    **0.0** and ``c`` to at most 9.8e-15, with the Hessian confirmed constant at
    two distinct points. The JAX branch is retained rather than deleted so an
    unrepresentable model still gets an answer.
    """
    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)

    terms = _qp_terms_tape(model, n_orig)
    if terms is None:
        terms = _qp_terms_jax(model, n_orig)
    Q, c_vec, obj_const = terms

    # Extract LP data for constraints (they're all linear)
    lp_data = extract_lp_data(model)
    n_slack = lp_data.c.shape[0] - n_orig

    # Extend Q with zeros for slack variables
    if n_slack > 0:
        n_total = n_orig + n_slack
        Q_full = np.zeros((n_total, n_total), dtype=np.float64)
        Q_full[:n_orig, :n_orig] = Q
        c_full = np.concatenate([c_vec, np.zeros(n_slack, dtype=np.float64)])
    else:
        Q_full = Q
        c_full = c_vec

    # Handle objective sense: negate for maximization (solvers always minimize)
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        Q_full = -Q_full
        c_full = -c_full
        obj_const = -obj_const

    return QPData(
        # numpy, not jnp: ``QPData`` is *annotated* for the JAX differentiation
        # consumers but populated by the JAX-free extractors (see the module
        # header). The ignores became necessary here only when this rung
        # stopped returning jnp
        # arrays, which is the point of the change.
        Q=Q_full,  # type: ignore[arg-type]
        c=c_full,  # type: ignore[arg-type]
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        x_l=lp_data.x_l,
        x_u=lp_data.x_u,
        obj_const=obj_const,
    )
