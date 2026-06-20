"""
AMPL .nl format writer for discopt models.

Produces text-mode .nl files compatible with AMPL-compatible solvers
(Ipopt, BARON, Couenne, SCIP, etc.) and the discopt Rust .nl parser.

The .nl format uses a prefix-notation expression encoding with numeric
opcodes. Key sections:

  Header (10 lines):  problem dimensions
  C sections:         nonlinear constraint bodies (expression DAG)
  O sections:         nonlinear objective (expression DAG)
  r section:          constraint bounds/senses
  b section:          variable bounds
  k section:          Jacobian column counts (cumulative)
  J sections:         linear terms in constraints (Jacobian)
  G sections:         linear terms in objective (gradient)
  x section:          initial point (optional)

Reference: Gay, D.M. "Hooking Your Solver to AMPL" (2003).
Inspired by Pyomo's NLv2Writer (pyomo/repn/plugins/nl_writer.py).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Union, cast

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    ObjectiveSense,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)


def to_nl(
    model: Model,
    path: Union[str, Path, None] = None,
) -> Union[str, None]:
    """Export a discopt Model to AMPL .nl text format.

    Parameters
    ----------
    model : Model
        A discopt optimization model.
    path : str or Path, optional
        If provided, write the .nl string to this file and return ``None``.
        Otherwise return the .nl string.

    Returns
    -------
    str or None
        The .nl text if *path* is ``None``, otherwise ``None``.
    """
    model.validate()
    writer = _NLWriter(model)
    text = writer.write()
    if path is not None:
        Path(path).write_text(text)
        return None
    return text


# ── Expression opcodes (AMPL .nl format) ────────────────────────
# Binary operators
_OP_ADD = 0
_OP_SUB = 1
_OP_MUL = 2
_OP_DIV = 3
_OP_POW = 5

# Unary operators
_OP_FLOOR = 13
_OP_CEIL = 14
_OP_ABS = 15
_OP_NEG = 16

# Math functions (Gay, "Writing .nl Files", 2005, Table 4)
_OP_TANH = 37
_OP_TAN = 38
_OP_SQRT = 39
_OP_SINH = 40
_OP_SIN = 41
_OP_LOG10 = 42
_OP_LOG = 43
_OP_EXP = 44
_OP_COSH = 45
_OP_COS = 46
_OP_ATANH = 47
_OP_ATAN = 49
_OP_ASINH = 50
_OP_ASIN = 51
_OP_ACOSH = 52
_OP_ACOS = 53
_OP_SUMLIST = 54

# Function name -> opcode mapping
_FUNC_OPCODES: dict[str, int] = {
    "exp": _OP_EXP,
    "log": _OP_LOG,
    "log10": _OP_LOG10,
    "sqrt": _OP_SQRT,
    "sin": _OP_SIN,
    "cos": _OP_COS,
    "asin": _OP_ASIN,
    "acos": _OP_ACOS,
    "atan": _OP_ATAN,
    "sinh": _OP_SINH,
    "cosh": _OP_COSH,
    "tanh": _OP_TANH,
    "abs": _OP_ABS,
}


class _NLWriter:
    def __init__(self, model: Model):
        self.model = model
        # Flatten all variables to a single indexed list
        # .nl format: continuous first, then binary, then integer (at end)
        self._flat_vars: list[tuple[Variable, int]] = []  # (var, element_idx)
        self._var_index: dict[tuple[str, int], int] = {}  # (name, elem) -> flat_idx
        self._n_total = 0
        # Separate linear and nonlinear parts
        self._obj_linear: dict[int, float] = {}  # var_idx -> coeff
        self._obj_nonlinear: Expression | None = None
        self._con_linear: list[dict[int, float]] = []  # per constraint
        self._con_nonlinear: list[Expression | None] = []
        self._con_bounds: list[tuple[int, float, float]] = []  # (type, lb, ub)
        # Header discrete/nonlinear counts, filled by _reorder_vars_canonical.
        self._nlvc_total = 0  # total nonlinear vars in constraints (incl. both)
        self._nlvo_total = 0  # total nonlinear vars in objectives (incl. both)
        self._nlvb = 0  # nonlinear vars in both cons + objs
        self._nlvbi = 0  # integer/binary vars nonlinear in both
        self._nlvci = 0  # integer/binary vars nonlinear in cons only
        self._nlvoi = 0  # integer/binary vars nonlinear in objs only
        self._nbv = 0  # linear binary vars
        self._niv = 0  # linear integer vars

    def write(self) -> str:
        self._build_var_map()
        self._decompose_expressions()
        self._reorder_vars_canonical()
        buf = io.StringIO()
        self._write_header(buf)
        self._write_C_sections(buf)
        self._write_O_section(buf)
        self._write_r_section(buf)
        self._write_b_section(buf)
        self._write_k_section(buf)
        self._write_J_sections(buf)
        self._write_G_section(buf)
        return buf.getvalue()

    # ── Build flat variable map ──

    def _build_var_map(self):
        """Build flat variable index map. Continuous vars first, then discrete."""
        continuous: list[tuple[Variable, int]] = []
        binary: list[tuple[Variable, int]] = []
        integer: list[tuple[Variable, int]] = []

        for var in self.model._variables:
            size = max(1, int(np.prod(var.shape)))
            for elem in range(size):
                entry = (var, elem)
                if var.var_type == VarType.BINARY:
                    binary.append(entry)
                elif var.var_type == VarType.INTEGER:
                    integer.append(entry)
                else:
                    continuous.append(entry)

        self._flat_vars = continuous + binary + integer
        self._n_total = len(self._flat_vars)
        self._n_binary = len(binary)
        self._n_integer = len(integer)

        for idx, (var, elem) in enumerate(self._flat_vars):
            self._var_index[(var.name, elem)] = idx

    # ── Reorder variables into canonical AMPL .nl order ──

    def _reorder_vars_canonical(self):
        """Reorder ``_flat_vars`` into the canonical AMPL ``.nl`` sequence.

        The ``.nl`` format requires nonlinear variables to occupy the lowest
        indices, grouped as ``[nl in both cons+objs | nl in cons only |
        nl in objs only | linear]``; within each nonlinear group the integer
        (and binary) members come **last**, and the linear tail is ordered
        ``[continuous | binary | integer]``. The header's discrete-count line
        (``nbv niv nlvbi nlvci nlvoi``) then declares how many of each group
        are discrete.

        The previous order (all continuous, then all discrete) wrote *every*
        discrete variable as linear-discrete (``nbv``/``niv``) with
        ``nlvbi=nlvci=nlvoi=0``. AMPL-compatible solvers (SCIP, BARON, Couenne,
        Ipopt) read a discrete variable appearing in a nonlinear expression as
        **continuous** under that encoding and silently solve a relaxation
        (issue #210). Pure MILP/LP export is unaffected — there the discrete
        vars are genuinely linear, so ``nbv``/``niv`` are already correct.

        Must run after :meth:`_decompose_expressions` (it needs the nonlinear
        expressions and the linear-coefficient dicts). It rebuilds
        ``_var_index`` and remaps the index-keyed ``_obj_linear`` /
        ``_con_linear`` dicts; nonlinear expressions resolve through
        ``_var_index`` at write time and so remap automatically.
        """
        # Classify variables (by current index) appearing in nonlinear parts.
        nl_cons: set[int] = set()
        nl_objs: set[int] = set()
        for nl in self._con_nonlinear:
            if nl is not None:
                self._collect_var_indices(nl, nl_cons)
        if self._obj_nonlinear is not None:
            self._collect_var_indices(self._obj_nonlinear, nl_objs)
        nl_both = nl_cons & nl_objs
        nl_cons_only = nl_cons - nl_both
        nl_objs_only = nl_objs - nl_both
        nl_all = nl_cons | nl_objs

        def is_discrete(old_idx: int) -> bool:
            var, _ = self._flat_vars[old_idx]
            return var.var_type in (VarType.BINARY, VarType.INTEGER)

        def split_group(idxs: set[int]) -> tuple[list[int], list[int]]:
            """Continuous members first, discrete members last (stable by index)."""
            ordered = sorted(idxs)
            cont = [i for i in ordered if not is_discrete(i)]
            disc = [i for i in ordered if is_discrete(i)]
            return cont, disc

        both_cont, both_disc = split_group(nl_both)
        cons_cont, cons_disc = split_group(nl_cons_only)
        objs_cont, objs_disc = split_group(nl_objs_only)

        # Linear tail: continuous, then binary, then integer (preserving order).
        lin_cont: list[int] = []
        lin_bin: list[int] = []
        lin_int: list[int] = []
        for old_idx, (var, _elem) in enumerate(self._flat_vars):
            if old_idx in nl_all:
                continue
            if var.var_type == VarType.BINARY:
                lin_bin.append(old_idx)
            elif var.var_type == VarType.INTEGER:
                lin_int.append(old_idx)
            else:
                lin_cont.append(old_idx)

        new_order = (
            both_cont
            + both_disc
            + cons_cont
            + cons_disc
            + objs_cont
            + objs_disc
            + lin_cont
            + lin_bin
            + lin_int
        )

        # Rebuild flat var list and the (name, elem) -> index map.
        old_flat = self._flat_vars
        self._flat_vars = [old_flat[i] for i in new_order]
        old2new = {old: new for new, old in enumerate(new_order)}
        self._var_index = {}
        for new_idx, (var, elem) in enumerate(self._flat_vars):
            self._var_index[(var.name, elem)] = new_idx

        # Remap index-keyed linear coefficient dicts.
        self._obj_linear = {old2new[k]: v for k, v in self._obj_linear.items()}
        self._con_linear = [{old2new[k]: v for k, v in lin.items()} for lin in self._con_linear]

        # Stash header counts (line 4: nlvc/nlvo totals + nlvb; line 6: discrete).
        self._nlvc_total = len(nl_cons)
        self._nlvo_total = len(nl_objs)
        self._nlvb = len(nl_both)
        self._nlvbi = len(both_disc)
        self._nlvci = len(cons_disc)
        self._nlvoi = len(objs_disc)
        self._nbv = len(lin_bin)
        self._niv = len(lin_int)

    # ── Decompose expressions into linear + nonlinear parts ──

    def _decompose_expressions(self):
        """Split objective and constraints into linear and nonlinear parts."""
        obj = self.model._objective
        if obj is not None:
            linear, nonlinear = self._split_expr(obj.expression)
            self._obj_linear = linear
            self._obj_nonlinear = nonlinear
        self._decompose_builder_objective()

        for con in self.model._constraints:
            # Determine constraint bound type (shared by every scalar row this
            # constraint expands into).
            rhs = con.rhs
            if con.sense == "<=":
                bnd = (1, 0.0, float(rhs))  # body <= rhs
            elif con.sense == ">=":
                bnd = (2, float(rhs), 0.0)  # body >= rhs (stored as <=)
            elif con.sense == "==":
                bnd = (4, float(rhs), float(rhs))
            else:
                bnd = (3, 0.0, 0.0)  # free (unreachable in practice)

            # Vectorized constraints (e.g. DAE/collocation bodies built from
            # matrix products and broadcasting) carry an array-valued body that
            # encodes many scalar constraints at once. Expand it into one scalar
            # expression per output element before linear/nonlinear splitting.
            bodies = self._scalarize_body(con.body)
            for body in bodies:
                linear, nonlinear = self._split_expr(body)
                self._con_linear.append(linear)
                self._con_nonlinear.append(nonlinear)
                self._con_bounds.append(bnd)

        self._decompose_builder_blocks()

    def _decompose_builder_objective(self):
        """Recover a linear or quadratic objective that lives only in the builder.

        ``add_linear_objective`` / ``add_quadratic_objective`` set the real
        ``0.5 x'Qx + c'x + constant`` objective in the builder and leave a zero
        placeholder in ``model._objective``. Reconstruct ``_obj_linear`` (the
        ``c'x`` gradient) and ``_obj_nonlinear`` (the quadratic part plus any
        constant offset) so ``.nl`` export carries the true objective. Only
        applies when the current objective is that placeholder, so a real
        expression objective set afterwards is never overwritten.
        """
        if not getattr(self.model._objective, "_is_placeholder", False):
            return
        lin_blk = getattr(self.model, "_builder_linear_objective", None)
        quad_blk = getattr(self.model, "_builder_quadratic_objective", None)
        if lin_blk is not None:
            c, x, constant, _sense = lin_blk
            self._obj_linear = self._linear_obj_from_c(c, x)
            self._obj_nonlinear = Constant(float(constant)) if constant != 0.0 else None
        elif quad_blk is not None:
            Q, c, x, constant, _sense = quad_blk
            self._obj_linear = self._linear_obj_from_c(c, x)
            self._obj_nonlinear = self._quadratic_obj_expr(Q, x, float(constant))

    def _linear_obj_from_c(self, c, x) -> dict[int, float]:
        """Map a cost vector over variable ``x`` to ``{global_var_index: coeff}``."""
        lin: dict[int, float] = {}
        for j, coeff in enumerate(np.asarray(c, dtype=np.float64).ravel()):
            coeff = float(coeff)
            if coeff == 0.0:
                continue
            gidx = self._var_index.get((x.name, j))
            if gidx is not None:
                lin[gidx] = lin.get(gidx, 0.0) + coeff
        return lin

    def _quadratic_obj_expr(self, Q, x, constant: float) -> Expression | None:
        """Build ``0.5 x'Sx + constant`` as an n-ary sum expression (or ``None``).

        The Rust builder reads only the **upper triangle** of ``Q`` and reflects
        it, i.e. it optimizes ``0.5 x'Sx`` with ``S = triu(Q) + striu(Q).T``
        (the strictly-lower triangle of the input is ignored). The export mirrors
        that exactly so it matches the solved model for any ``Q`` — symmetric,
        triangular, or asymmetric. Each stored entry ``S[i, j]`` contributes
        ``0.5 S[i, j] x[i] x[j]`` (diagonals give the ``x[i]^2`` terms). The sum
        is emitted as a single ``SUMLIST`` node so the writer never recurses
        through a deep ``+`` chain.
        """
        import scipy.sparse as sp

        S = (sp.triu(Q, 0) + sp.triu(Q, 1).T).tocsr()
        indptr = S.indptr
        indices = S.indices
        data = S.data
        terms: list[Expression] = []
        for i in range(S.shape[0]):
            for k in range(int(indptr[i]), int(indptr[i + 1])):
                q = float(data[k])
                if q == 0.0:
                    continue
                j = int(indices[k])
                prod = BinaryOp("*", IndexExpression(x, i), IndexExpression(x, j))
                terms.append(BinaryOp("*", Constant(0.5 * q), prod))
        if constant != 0.0:
            terms.append(Constant(constant))
        if not terms:
            return None
        if len(terms) == 1:
            return terms[0]
        return SumOverExpression(terms)

    def _decompose_builder_blocks(self):
        """Emit linear constraints that live only in the Rust builder.

        The fast-construction API (``add_linear_constraints``) and the indexed
        constraint fast path build rows directly into the builder, bypassing
        ``model._constraints``. Each recorded ``(A, x, sense, b, name)`` block is
        reconstructed here as purely linear rows so ``.nl`` export sees the same
        model the solver does.
        """
        blocks = getattr(self.model, "_builder_linear_blocks", None)
        if not blocks:
            return
        for A, x, sense, b, _name in blocks:
            indptr = A.indptr
            indices = A.indices
            data = A.data
            for r in range(A.shape[0]):
                lin: dict[int, float] = {}
                for k in range(int(indptr[r]), int(indptr[r + 1])):
                    coeff = float(data[k])
                    if coeff == 0.0:
                        continue
                    gidx = self._var_index.get((x.name, int(indices[k])))
                    if gidx is not None:
                        lin[gidx] = lin.get(gidx, 0.0) + coeff
                rhs = float(b[r])
                if sense == "<=":
                    bnd = (1, 0.0, rhs)  # A x <= b
                elif sense == ">=":
                    bnd = (2, rhs, 0.0)  # A x >= b
                elif sense == "==":
                    bnd = (4, rhs, rhs)
                else:  # pragma: no cover - sense is validated upstream
                    bnd = (3, 0.0, 0.0)
                self._con_linear.append(lin)
                self._con_nonlinear.append(None)
                self._con_bounds.append(bnd)

    def _scalarize_body(self, expr: Expression) -> list[Expression]:
        """Return the scalar constraint bodies a (possibly array) body expands to.

        A scalar body yields a single-element list; an array body yields one
        scalar expression per element in row-major order.

        A purely scalar body (the common case, including deep ``sum()`` chains)
        skips the recursive :meth:`_scalarize` pass entirely — the downstream
        linear/nonlinear split and writer already traverse scalar nodes
        iteratively, so this avoids a recursion-limit overflow on deep DAGs.
        Only genuinely array-structured bodies (matrix products, axis
        reductions, unindexed array variables) need expansion.
        """
        if not self._needs_scalarize(expr):
            return [expr]
        arr = self._scalarize(expr)
        if arr.ndim == 0:
            return [cast(Expression, arr[()])]
        return list(arr.ravel())

    def _needs_scalarize(self, expr: Expression) -> bool:
        """True if ``expr`` has array structure requiring element expansion.

        Walks the DAG iteratively (no recursion). Returns ``True`` on the first
        node that introduces array shape — an unindexed array ``Variable``, a
        ``MatMulExpression``, an axis-reducing ``SumExpression``, or an
        ``IndexExpression`` that does not resolve to a single scalar variable.
        Scalar-only bodies return ``False`` and bypass :meth:`_scalarize`.
        """
        stack: list[Expression] = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Variable):
                if node.shape not in ((), (1,)):
                    return True
            elif isinstance(node, (MatMulExpression, SumExpression)):
                return True
            elif isinstance(node, IndexExpression):
                if self._resolve_var_index(node) is None:
                    return True
            elif isinstance(node, BinaryOp):
                stack.append(node.left)
                stack.append(node.right)
            elif isinstance(node, UnaryOp):
                stack.append(node.operand)
            elif isinstance(node, FunctionCall):
                stack.extend(node.args)
            elif isinstance(node, SumOverExpression):
                stack.extend(node.terms)
            # Constant, scalar Variable, resolved IndexExpression, opaque leaves: fine
        return False

    @staticmethod
    def _obj0(x) -> np.ndarray:
        """Wrap a single expression in a 0-d object array."""
        out = np.empty((), dtype=object)
        out[()] = x
        return out

    def _sum_terms(self, terms: list[Expression]) -> Expression:
        """Left-fold a list of scalar expressions into a sum (``+``)."""
        if not terms:
            return Constant(0.0)
        result: Expression = terms[0]
        for t in terms[1:]:
            result = BinaryOp("+", result, t)
        return result

    def _matmul_scalar(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Symbolic matmul of two object arrays of scalar expressions."""
        if left.ndim == 1 and right.ndim == 1:
            (k,) = left.shape
            return self._obj0(self._sum_terms([BinaryOp("*", left[i], right[i]) for i in range(k)]))
        if left.ndim == 2 and right.ndim == 1:
            m, k = left.shape
            out = np.empty((m,), dtype=object)
            for i in range(m):
                out[i] = self._sum_terms([BinaryOp("*", left[i, p], right[p]) for p in range(k)])
            return out
        if left.ndim == 1 and right.ndim == 2:
            k, n = right.shape
            out = np.empty((n,), dtype=object)
            for j in range(n):
                out[j] = self._sum_terms([BinaryOp("*", left[p], right[p, j]) for p in range(k)])
            return out
        if left.ndim == 2 and right.ndim == 2:
            m, k = left.shape
            _, n = right.shape
            out = np.empty((m, n), dtype=object)
            for i in range(m):
                for j in range(n):
                    out[i, j] = self._sum_terms(
                        [BinaryOp("*", left[i, p], right[p, j]) for p in range(k)]
                    )
            return out
        raise ValueError(f"Unsupported matmul of shapes {left.shape} @ {right.shape}")

    def _sum_axis(self, arr: np.ndarray, axis: int | None) -> np.ndarray:
        """Symbolic reduction (``+``) of an object array along ``axis``."""
        if axis is None:
            return self._obj0(self._sum_terms([arr[idx] for idx in np.ndindex(arr.shape)]))
        moved = np.moveaxis(arr, axis, 0)
        out_shape = moved.shape[1:]
        out = np.empty(out_shape, dtype=object)
        for idx in np.ndindex(out_shape):
            out[idx] = self._sum_terms([moved[(p, *idx)] for p in range(moved.shape[0])])
        if out.ndim == 0:
            return self._obj0(out[()])
        return out

    def _scalarize(self, expr: Expression) -> np.ndarray:
        """Expand an expression into an object ndarray of scalar expressions.

        Indexing, broadcasting, and matrix products are pushed through the DAG
        so each output element becomes a plain scalar expression that the rest
        of the (scalar-oriented) writer can decompose. Scalars are returned as
        0-d object arrays. Node types without a known array structure (e.g.
        parameters) are treated as opaque scalar leaves.
        """
        if isinstance(expr, Constant):
            v = expr.value
            if v.ndim == 0:
                return self._obj0(Constant(float(v)))
            out = np.empty(v.shape, dtype=object)
            for idx in np.ndindex(v.shape):
                out[idx] = Constant(float(v[idx]))
            return out

        if isinstance(expr, Variable):
            if expr.shape == () or expr.shape == (1,):
                return self._obj0(expr)
            out = np.empty(expr.shape, dtype=object)
            ndim = len(expr.shape)
            for idx in np.ndindex(expr.shape):
                out[idx] = IndexExpression(expr, idx if ndim > 1 else idx[0])
            return out

        if isinstance(expr, IndexExpression):
            base = self._scalarize(expr.base)
            sub = base[expr.index]
            return sub if isinstance(sub, np.ndarray) else self._obj0(sub)

        if isinstance(expr, UnaryOp):
            operand = self._scalarize(expr.operand)
            out = np.empty(operand.shape, dtype=object)
            for idx in np.ndindex(operand.shape):
                out[idx] = UnaryOp(expr.op, operand[idx])
            return out

        if isinstance(expr, BinaryOp):
            left, right = np.broadcast_arrays(
                self._scalarize(expr.left), self._scalarize(expr.right)
            )
            out = np.empty(left.shape, dtype=object)
            for idx in np.ndindex(left.shape):
                out[idx] = BinaryOp(expr.op, left[idx], right[idx])
            return out

        if isinstance(expr, FunctionCall):
            args = [self._scalarize(a) for a in expr.args]
            bargs = list(np.broadcast_arrays(*args)) if len(args) > 1 else args
            shape = bargs[0].shape
            out = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                out[idx] = FunctionCall(expr.func_name, *[b[idx] for b in bargs])
            return out

        if isinstance(expr, MatMulExpression):
            return self._matmul_scalar(self._scalarize(expr.left), self._scalarize(expr.right))

        if isinstance(expr, SumExpression):
            return self._sum_axis(self._scalarize(expr.operand), expr.axis)

        if isinstance(expr, SumOverExpression):
            terms = [self._scalarize(t) for t in expr.terms]
            bterms = list(np.broadcast_arrays(*terms)) if len(terms) > 1 else terms
            shape = bterms[0].shape
            out = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                out[idx] = self._sum_terms([b[idx] for b in bterms])
            return out

        # Opaque scalar leaf (e.g. Parameter): leave intact.
        return self._obj0(expr)

    def _split_expr(self, expr: Expression) -> tuple[dict[int, float], Expression | None]:
        """Split an expression into linear terms {var_idx: coeff} and nonlinear remainder."""
        linear: dict[int, float] = {}
        nonlinear_terms: list[Expression] = []
        self._collect_linear(expr, 1.0, linear, nonlinear_terms)
        if nonlinear_terms:
            if len(nonlinear_terms) == 1:
                nl_expr = nonlinear_terms[0]
            else:
                nl_expr = nonlinear_terms[0]
                for t in nonlinear_terms[1:]:
                    nl_expr = BinaryOp("+", nl_expr, t)
            return linear, nl_expr
        return linear, None

    def _collect_linear(
        self,
        expr: Expression,
        coeff: float,
        linear: dict[int, float],
        nonlinear: list[Expression],
    ):
        """Collect linear terms ``{var_idx: coeff}`` and segregate nonlinear ones.

        Driven by an explicit ``(expr, coeff)`` work stack rather than recursion:
        a body built with ``sum()`` of many terms is a deeply left-nested chain
        of ``BinaryOp`` "+" nodes, which overflows Python's recursion limit. The
        ``nonlinear`` output list is order-sensitive (it is later folded into a
        sum), so children are pushed reversed to preserve the original
        left-to-right pre-order in which terms were appended.
        """
        stack: list[tuple[Expression, float]] = [(expr, coeff)]
        while stack:
            node, c = stack.pop()

            if isinstance(node, Constant):
                val = float(node.value)
                if val != 0.0:
                    # A nonzero constant contributes a constant offset.
                    nonlinear.append(Constant(val * c))
                continue

            if isinstance(node, Variable):
                if node.shape == () or node.shape == (1,):
                    idx = self._var_index.get((node.name, 0))
                    if idx is not None:
                        linear[idx] = linear.get(idx, 0.0) + c
                        continue
                # Array variable without indexing - treat as nonlinear
                nonlinear.append(node if c == 1.0 else BinaryOp("*", Constant(c), node))
                continue

            if isinstance(node, IndexExpression):
                var_idx = self._resolve_var_index(node)
                if var_idx is not None:
                    linear[var_idx] = linear.get(var_idx, 0.0) + c
                    continue
                nonlinear.append(node if c == 1.0 else BinaryOp("*", Constant(c), node))
                continue

            if isinstance(node, BinaryOp):
                if node.op == "+":
                    stack.append((node.right, c))
                    stack.append((node.left, c))
                    continue
                if node.op == "-":
                    stack.append((node.right, -c))
                    stack.append((node.left, c))
                    continue
                if node.op == "*":
                    # Fold in a constant factor on either side.
                    lc = self._get_const(node.left)
                    rc = self._get_const(node.right)
                    if lc is not None:
                        stack.append((node.right, c * lc))
                        continue
                    if rc is not None:
                        stack.append((node.left, c * rc))
                        continue
                # Not linear (/, **, or product of two non-constants)
                nonlinear.append(node if c == 1.0 else BinaryOp("*", Constant(c), node))
                continue

            if isinstance(node, UnaryOp) and node.op == "neg":
                stack.append((node.operand, -c))
                continue

            if isinstance(node, SumOverExpression):
                for term in reversed(node.terms):
                    stack.append((term, c))
                continue

            # Everything else is nonlinear
            nonlinear.append(node if c == 1.0 else BinaryOp("*", Constant(c), node))

    def _get_const(self, expr: Expression) -> float | None:
        # Only 0-d (scalar) constants are linear coefficients. Array-valued
        # constants are pushed to scalars by _scalarize before we get here;
        # guard defensively so a stray array constant degrades to "nonlinear"
        # rather than raising in float().
        if isinstance(expr, Constant) and expr.value.ndim == 0:
            return float(expr.value)
        return None

    def _resolve_var_index(self, expr: IndexExpression) -> int | None:
        """Resolve x[i] or x[i,j] to a flat variable index."""
        if not isinstance(expr.base, Variable):
            return None
        var = expr.base
        idx = expr.index
        if isinstance(idx, tuple):
            # Multi-dimensional: flatten to row-major
            flat = 0
            for dim, i in enumerate(idx):
                if not isinstance(i, int):
                    return None
                stride = 1
                for d2 in range(dim + 1, len(var.shape)):
                    stride *= var.shape[d2]
                flat += i * stride
            return self._var_index.get((var.name, flat))
        if isinstance(idx, int):
            return self._var_index.get((var.name, idx))
        return None

    # ── Header (10 mandatory lines) ──

    def _write_header(self, buf: io.StringIO):
        n_vars = self._n_total
        # Use the expanded scalar-constraint count: a single vectorized
        # (DAE/collocation) constraint expands into many scalar rows.
        n_cons = len(self._con_linear)
        n_objs = 1 if self.model._objective else 0

        # Count nonlinear constraints and objectives
        n_nl_cons = sum(1 for nl in self._con_nonlinear if nl is not None)
        n_nl_objs = 1 if self._obj_nonlinear is not None else 0

        # Count Jacobian nonzeros (linear + nonlinear vars in each constraint)
        n_jac_nz = sum(len(lin) for lin in self._con_linear)
        # Add nonlinear variable references in constraints
        for nl in self._con_nonlinear:
            if nl is not None:
                vs: set[int] = set()
                self._collect_var_indices(nl, vs)
                n_jac_nz += len(vs)

        # Count objective gradient nonzeros
        n_grad_nz = len(self._obj_linear)
        if self._obj_nonlinear is not None:
            vs2: set[int] = set()
            self._collect_var_indices(self._obj_nonlinear, vs2)
            n_grad_nz += len(vs2)

        # Line 0: format marker
        buf.write(f"g3 1 1 0\t# problem {self.model.name}\n")
        # Line 1: core dimensions
        buf.write(f" {n_vars} {n_cons} {n_objs} 0 0\t# vars, constraints, objectives\n")
        # Line 2: nonlinearity counts
        buf.write(f" {n_nl_cons} {n_nl_objs}\t# nonlinear constraints, objectives\n")
        # Line 3: network (unused)
        buf.write(" 0 0\t# network constraints\n")
        # Line 4: nonlinear variable distribution (nlvc/nlvo are TOTALS, incl. both)
        buf.write(
            f" {self._nlvc_total} {self._nlvo_total} {self._nlvb}"
            "\t# nonlinear vars in cons, objs, both\n"
        )
        # Line 5: flags
        buf.write(" 0 0 0 1 0\t# flags\n")
        # Line 6: discrete variable counts: nbv niv nlvbi nlvci nlvoi
        # (linear-binary, linear-integer, then integer/binary vars nonlinear in
        #  both / cons-only / objs-only — see _reorder_vars_canonical, issue #210)
        buf.write(
            f" {self._nbv} {self._niv} {self._nlvbi} {self._nlvci} {self._nlvoi}"
            "\t# nbv niv nlvbi nlvci nlvoi\n"
        )
        # Line 7: sparsity
        buf.write(f" {n_jac_nz} {n_grad_nz}\t# Jacobian, gradient nonzeros\n")
        # Line 8: name lengths
        buf.write(" 0 0\t# max name lengths\n")
        # Line 9: common expressions
        buf.write(" 0 0 0 0 0\t# common expressions\n")

    def _collect_var_indices(self, expr: Expression, result: set[int]):
        """Collect all variable indices referenced in an expression.

        Uses an explicit work stack rather than recursion: a model built with
        ``sum()`` of many terms produces a deeply left-nested chain of
        ``BinaryOp`` nodes, and recursing over it overflows Python's recursion
        limit. Traversal order is irrelevant since ``result`` is a set.
        """
        stack: list[Expression] = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Variable):
                if node.shape == () or node.shape == (1,):
                    idx = self._var_index.get((node.name, 0))
                    if idx is not None:
                        result.add(idx)
            elif isinstance(node, IndexExpression):
                vi = self._resolve_var_index(node)
                if vi is not None:
                    result.add(vi)
            elif isinstance(node, BinaryOp):
                stack.append(node.left)
                stack.append(node.right)
            elif isinstance(node, UnaryOp):
                stack.append(node.operand)
            elif isinstance(node, FunctionCall):
                stack.extend(node.args)
            elif isinstance(node, SumExpression):
                stack.append(node.operand)
            elif isinstance(node, SumOverExpression):
                stack.extend(node.terms)

    # ── C sections (nonlinear constraint bodies) ──

    def _write_C_sections(self, buf: io.StringIO):
        for i, nl in enumerate(self._con_nonlinear):
            buf.write(f"C{i}\n")
            if nl is not None:
                self._write_expr(nl, buf)
            else:
                buf.write("n0\n")

    # ── O section (nonlinear objective) ──

    def _write_O_section(self, buf: io.StringIO):
        if self.model._objective is None:
            return
        sense = 0 if self.model._objective.sense == ObjectiveSense.MINIMIZE else 1
        buf.write(f"O0 {sense}\n")
        if self._obj_nonlinear is not None:
            self._write_expr(self._obj_nonlinear, buf)
        else:
            buf.write("n0\n")

    # ── r section (constraint bounds) ──

    def _write_r_section(self, buf: io.StringIO):
        if not self._con_bounds:
            return
        buf.write("r\n")
        for btype, lb, ub in self._con_bounds:
            if btype == 1:  # <= ub
                buf.write(f"1 {ub}\n")
            elif btype == 2:  # >= lb
                buf.write(f"2 {lb}\n")
            elif btype == 4:  # == rhs
                buf.write(f"4 {lb}\n")
            elif btype == 0:  # range
                buf.write(f"0 {lb} {ub}\n")
            else:
                buf.write("3\n")  # free

    # ── b section (variable bounds) ──

    def _write_b_section(self, buf: io.StringIO):
        buf.write("b\n")
        for var, elem in self._flat_vars:
            lb_arr = np.asarray(var.lb).flat
            ub_arr = np.asarray(var.ub).flat
            lb = float(lb_arr[elem]) if elem < len(lb_arr) else float(lb_arr[0])
            ub = float(ub_arr[elem]) if elem < len(ub_arr) else float(ub_arr[0])

            has_lb = lb > -1e18
            has_ub = ub < 1e18

            if has_lb and has_ub:
                if abs(lb - ub) < 1e-15:
                    buf.write(f"4 {lb}\n")  # fixed
                else:
                    buf.write(f"0 {lb} {ub}\n")  # range
            elif has_lb:
                buf.write(f"2 {lb}\n")  # lb only
            elif has_ub:
                buf.write(f"1 {ub}\n")  # ub only
            else:
                buf.write("3\n")  # free

    # ── k section (Jacobian column counts) ──

    def _write_k_section(self, buf: io.StringIO):
        if self._n_total <= 1:
            buf.write("k0\n")
            return
        buf.write(f"k{self._n_total - 1}\n")
        cumulative = 0
        for col in range(self._n_total - 1):
            count = 0
            for lin in self._con_linear:
                if col in lin:
                    count += 1
            cumulative += count
            buf.write(f"{cumulative}\n")

    # ── J sections (linear Jacobian terms) ──

    def _write_J_sections(self, buf: io.StringIO):
        for i, lin in enumerate(self._con_linear):
            if not lin:
                continue
            buf.write(f"J{i} {len(lin)}\n")
            for var_idx in sorted(lin.keys()):
                buf.write(f"{var_idx} {lin[var_idx]}\n")

    # ── G section (linear objective gradient) ──

    def _write_G_section(self, buf: io.StringIO):
        if not self._obj_linear:
            return
        buf.write(f"G0 {len(self._obj_linear)}\n")
        for var_idx in sorted(self._obj_linear.keys()):
            buf.write(f"{var_idx} {self._obj_linear[var_idx]}\n")

    # ── Expression → .nl opcode encoding ──

    def _write_expr(self, expr: Expression, buf: io.StringIO):
        """Emit an expression in .nl prefix notation.

        Driven by an explicit work stack rather than recursion so deeply
        left-nested DAGs (``sum()`` of many terms builds a long chain of
        ``BinaryOp`` "+" nodes) cannot exceed Python's recursion limit. Stack
        items are either ``Expression`` nodes (to encode) or pre-rendered token
        strings (written verbatim); children are pushed in reverse so they pop
        in left-to-right prefix order.
        """
        stack: list = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, str):
                buf.write(node)
                continue

            if isinstance(node, Constant):
                buf.write(f"n{float(node.value)}\n")
            elif isinstance(node, Variable):
                if node.shape == () or node.shape == (1,):
                    idx = self._var_index.get((node.name, 0))
                    if idx is not None:
                        buf.write(f"v{idx}\n")
                        continue
                raise ValueError(f"Cannot write array variable {node.name} without indexing")
            elif isinstance(node, IndexExpression):
                vi = self._resolve_var_index(node)
                if vi is None:
                    raise ValueError(f"Cannot resolve indexed expression: {node}")
                buf.write(f"v{vi}\n")
            elif isinstance(node, BinaryOp):
                op_map = {
                    "+": _OP_ADD,
                    "-": _OP_SUB,
                    "*": _OP_MUL,
                    "/": _OP_DIV,
                    "**": _OP_POW,
                }
                opcode = op_map.get(node.op)
                if opcode is None:
                    raise ValueError(f"Unknown binary operator: {node.op}")
                buf.write(f"o{opcode}\n")
                stack.append(node.right)
                stack.append(node.left)
            elif isinstance(node, UnaryOp):
                if node.op == "neg":
                    buf.write(f"o{_OP_NEG}\n")
                elif node.op == "abs":
                    buf.write(f"o{_OP_ABS}\n")
                else:
                    raise ValueError(f"Unknown unary operator: {node.op}")
                stack.append(node.operand)
            elif isinstance(node, FunctionCall):
                # Build the emit sequence (tokens + sub-expressions) and push it
                # reversed; tokens are written when popped.
                seq = self._function_call_sequence(node)
                stack.extend(reversed(seq))
            elif isinstance(node, SumExpression):
                stack.append(node.operand)
            elif isinstance(node, SumOverExpression):
                if len(node.terms) == 0:
                    buf.write("n0\n")
                elif len(node.terms) == 1:
                    stack.append(node.terms[0])
                else:
                    buf.write(f"o{_OP_SUMLIST}\n")
                    buf.write(f"{len(node.terms)}\n")
                    stack.extend(reversed(node.terms))
            elif isinstance(node, MatMulExpression):
                raise ValueError(
                    "MatMul expressions must be expanded before .nl export. "
                    "Use explicit indexing instead of @."
                )
            elif isinstance(node, CustomCall):
                raise ValueError(
                    f"Cannot export the opaque AD-only user function {node.name!r} "
                    f"(dm.custom) to .nl: it has no AMPL opcode. Rebuild the function "
                    f"from dm.* primitives and use dm.udf to make it exportable."
                )
            else:
                raise ValueError(f"Cannot write expression type to .nl: {type(node).__name__}")

    def _function_call_sequence(self, expr: FunctionCall) -> list:
        """Return the ordered .nl emit sequence for a function call.

        Each element is either a token string (written verbatim) or an
        ``Expression`` to be encoded. Returned in prefix order; the caller
        pushes it reversed onto the work stack.
        """
        fname = expr.func_name.lower()
        if fname == "tan":
            # tan(x) = sin(x) / cos(x)
            return [f"o{_OP_DIV}\n", f"o{_OP_SIN}\n", expr.args[0], f"o{_OP_COS}\n", expr.args[0]]
        if fname == "log2":
            # log2(x) = log(x) / log(2)
            return [f"o{_OP_DIV}\n", f"o{_OP_LOG}\n", expr.args[0], f"n{math.log(2)}\n"]
        if fname == "log1p":
            # log1p(x) = log(1 + x)
            return [f"o{_OP_LOG}\n", f"o{_OP_ADD}\n", "n1\n", expr.args[0]]
        if fname == "sigmoid":
            # sigmoid(x) = 1 / (1 + exp(-x))
            return [
                f"o{_OP_DIV}\n",
                "n1\n",
                f"o{_OP_ADD}\n",
                "n1\n",
                f"o{_OP_EXP}\n",
                f"o{_OP_NEG}\n",
                expr.args[0],
            ]
        if fname == "softplus":
            # softplus(x) = log(1 + exp(x))
            return [f"o{_OP_LOG}\n", f"o{_OP_ADD}\n", "n1\n", f"o{_OP_EXP}\n", expr.args[0]]
        if fname == "erf":
            raise ValueError("erf() has no .nl opcode; reformulate without erf")
        if fname == "sign":
            raise ValueError("sign() has no .nl opcode; reformulate without sign")
        if fname in ("min", "max"):
            raise ValueError(f"{fname}() requires DNLP model type; not supported in .nl export")
        opcode = _FUNC_OPCODES.get(fname)
        if opcode is not None:
            return [f"o{opcode}\n", expr.args[0]]
        raise ValueError(f"Unknown function in .nl export: {fname}")
