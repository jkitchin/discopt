"""Exact row and objective evaluation of an all-affine model, with no evaluator build.

#1743. After the HiGHS LP route has solved a model, four consumers re-read it:
dual recovery (``solver._matched_row_constraint_duals``), the certified-incumbent
backstop (#1561), incumbent repair (#1537 E) and objective reconciliation. The
last three used to build a general NLP evaluator -- the POUNCE tape, or, when the
model is past the tape's node cap, the JAX evaluator, whose trace over a 1M-term
per-element transport LP did not finish in 700 s after HiGHS had solved it.

They need very little of it: row values, the row Jacobian, the row map and the
objective value. For a model whose objective and constraints are all affine, every
one of those is exact from :func:`discopt._relax.affine_array.affine_of`: ``J`` and
``g(0)`` in the evaluators' own row layout, and the objective as ``c x + c0``.
That costs O(nnz) and no tape or trace.

Nothing here is an approximation of the evaluator it replaces: the Jacobian of an
affine map is constant, the row map is the same lazy constraints in the same
order, and the values differ from a tape's only by summation order (round-off).
A model that is not all-affine is declined (``None``) and the caller builds the
evaluator as before.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from discopt.modeling.core import Constraint, Model, ObjectiveSense


def _csr_rows(A: Any, n_cols: int) -> Any:
    """``A`` as canonical CSR (sorted, duplicates summed), at most ``n_cols`` columns."""
    M = sp.csr_matrix(A, dtype=np.float64)
    if M.shape[1] > n_cols:
        M = M[:, :n_cols].tocsr()
    M.sum_duplicates()
    M.sort_indices()
    return M


def affine_constraint_rows(
    model: Model, n_orig: int
) -> Optional[tuple[Any, np.ndarray, list[tuple[int, int, Any]]]]:
    """``(J, g(0), row_map)`` of an all-affine constraint set, or ``None`` (#1720).

    Read off :func:`discopt._relax.affine_array.affine_of`: exact, sparse, O(nnz),
    and in the evaluator's row layout -- one row per element of each body in C
    order, constraints in ``model._constraints`` order -- so ``row_map`` has the
    shape :meth:`constraint_row_map` returns. ``None`` when any body is not affine;
    the caller then uses an evaluator.

    Builder-resident rows (#840: ``add_linear_constraints`` and the
    ``Model.constraint`` linear fast path) follow, one scalar row each, read off
    their retained CSR blocks (#1743). Before, any such row declined the whole
    walk, and a per-element 1000x1000 transport LP -- past the tape's node cap --
    fell through to a JAX evaluator trace over 1M terms after HiGHS had solved it.
    The rows are named by the same lazy constraints both evaluators list after
    ``model._constraints``, and each body is the one
    :func:`discopt.modeling.core._builder_row_constraint` builds: ``a x - b`` for
    ``<=``/``==``, ``b - a x`` (sense ``<=``) for ``>=``.
    """
    from discopt._relax.affine_array import NotAffineError, affine_of

    blocks = []
    row_map: list[tuple[int, int, Any]] = []
    start = 0
    try:
        for con in model._constraints:
            if not isinstance(con, Constraint):
                continue
            aff = affine_of(con.body, model._flat_var_offset, n_orig)
            M = aff.M if aff.M is not None else sp.csr_matrix((aff.size, n_orig))
            blocks.append((M, aff.b))
            row_map.append((start, start + aff.size, con))
            start += aff.size
    except NotAffineError:
        return None
    builder_blocks = getattr(model, "_builder_linear_blocks", None)
    if builder_blocks:
        lazy_rows = [
            c for c in model._builder_linear_constraints(lazy=True) if isinstance(c, Constraint)
        ]
        r = 0
        for A, x, sense, b, _name in builder_blocks:
            A = sp.csr_matrix(A, dtype=np.float64)
            k, n_local = int(A.shape[0]), int(A.shape[1])
            off = int(model._flat_var_offset(x))
            if off < 0 or off + n_local > n_orig or sense not in ("<=", ">=", "=="):
                return None
            M = sp.csr_matrix((A.data, A.indices + off, A.indptr), shape=(k, n_orig))
            rhs = np.broadcast_to(np.asarray(b, dtype=np.float64), (k,)).copy()
            if sense == ">=":
                M, g = -M, rhs
            else:
                g = -rhs
            blocks.append((M.tocsr(), g))
            for con in lazy_rows[r : r + k]:
                row_map.append((start, start + 1, con))
                start += 1
            r += k
        if r != len(lazy_rows):
            return None
    if not blocks:
        return _csr_rows(sp.csr_matrix((0, n_orig)), n_orig), np.zeros(0), row_map
    J = _csr_rows(sp.vstack([M for M, _ in blocks], format="csr"), n_orig)
    g0 = np.concatenate([b for _, b in blocks]).astype(np.float64)
    return J, g0, row_map


class AffineEvaluator:
    """The evaluator protocol's row and objective half, exact for an all-affine model.

    Implements what the post-solve consumers read -- ``n_variables``,
    ``n_constraints``, ``variable_bounds``, ``constraint_row_map``,
    ``evaluate_objective`` / ``evaluate_gradient`` (minimized, negated for a
    MAXIMIZE model, as the tape and JAX evaluators report it),
    ``evaluate_constraints`` and the Jacobian in sparse and dense form -- and
    nothing else: no Hessian, so an NLP solver handed one fails loudly rather than
    getting a guessed one. Build with :meth:`try_build`.
    """

    def __init__(
        self,
        model: Model,
        J: Any,
        g0: np.ndarray,
        row_map: list[tuple[int, int, Any]],
        c: Any,
        c0: float,
    ) -> None:
        if model._objective is None:
            raise ValueError("AffineEvaluator needs a model with an objective")
        self._model = model
        self._negate = model._objective.sense == ObjectiveSense.MAXIMIZE
        self._J = J
        self._g0 = g0
        self._row_map = row_map
        self._c = c  # (1, n) CSR, or None for a constant objective
        self._c0 = float(c0)
        self._n_variables = int(J.shape[1])

    @classmethod
    def try_build(cls, model: Model) -> Optional["AffineEvaluator"]:
        """The evaluator when ``model``'s objective and every constraint are affine.

        Gated by the LP/QP family test the incumbent-verification snapshot already
        uses (``_is_fast_linear_quadratic_family``) and a degree-1 objective, both
        pure-Python degree walks, so a nonlinear model is declined before any
        matrix is assembled.
        """
        from discopt._relax.affine_array import NotAffineError, affine_of
        from discopt.modeling.core import _expression_degree, _is_fast_linear_quadratic_family

        if model._objective is None or not _is_fast_linear_quadratic_family(model):
            return None
        obj = model._objective_expression()
        if _expression_degree(obj) > 1:
            return None
        n = int(sum(int(v.size) for v in model._variables))
        rows = affine_constraint_rows(model, n)
        if rows is None:
            return None
        try:
            aff = affine_of(obj, model._flat_var_offset, n)
        except NotAffineError:
            return None
        if aff.size != 1:
            return None
        J, g0, row_map = rows
        c = None if aff.M is None else _csr_rows(aff.M, n)
        return cls(model, J, g0, row_map, c, float(aff.b[0]))

    @property
    def n_variables(self) -> int:
        return self._n_variables

    @property
    def n_constraints(self) -> int:
        return int(self._J.shape[0])

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lbs, ubs = [], []
        for v in self._model._variables:
            lbs.append(np.asarray(v.lb, dtype=np.float64).flatten())
            ubs.append(np.asarray(v.ub, dtype=np.float64).flatten())
        if not lbs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lbs), np.concatenate(ubs)

    def constraint_row_map(self) -> list[tuple[int, int, Any]]:
        return list(self._row_map)

    def _x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self._n_variables:
            raise ValueError(f"point has {x.shape[0]} entries, model has {self._n_variables}")
        return x

    def evaluate_objective(self, x: np.ndarray) -> float:
        x = self._x(x)
        f = self._c0 + (0.0 if self._c is None else float((self._c @ x)[0]))
        return -f if self._negate else f

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        self._x(x)
        if self._c is None:
            grad = np.zeros(self._n_variables, dtype=np.float64)
        else:
            grad = np.asarray(self._c.toarray(), dtype=np.float64).ravel()
        return -grad if self._negate else grad

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        g: np.ndarray = np.asarray(self._J @ self._x(x), dtype=np.float64) + self._g0
        return g

    def evaluate_sparse_jacobian(self, x: np.ndarray) -> Any:
        self._x(x)
        return self._J.copy()

    def evaluate_jacobian(self, x: np.ndarray) -> np.ndarray:
        self._x(x)
        return np.asarray(self._J.toarray(), dtype=np.float64)


def post_solve_evaluator(model: Model) -> Any:
    """:class:`AffineEvaluator` for an all-affine model, else ``make_evaluator``.

    For the post-solve consumers that read only rows and the objective (#1743).
    """
    ev = AffineEvaluator.try_build(model)
    if ev is not None:
        return ev
    from discopt._tape_nlp_evaluator import make_evaluator

    return make_evaluator(model)
