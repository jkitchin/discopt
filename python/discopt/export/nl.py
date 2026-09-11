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
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np

from discopt.export import _arrays
from discopt.export._common import refuse_non_algebraic_relations
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
    Parameter,
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
    # Ahead of the Rust fast path on purpose. `_NLWriter.write()` makes this same
    # call (#1218), but `to_nl` only reaches that writer when `_rust_nl_text`
    # declines -- so relying on it would make the refusal depend on whether the
    # fast path happened to choke on a disjunctive row, rather than on the model.
    refuse_non_algebraic_relations(model, ".nl")

    # ``for_solve=False``: this writer *honours* ``Constraint.rhs`` — it folds the
    # body constant into the r-section row bound — so a row the solve path refuses
    # as unrepresentable (#909) is still exported faithfully here. Refusing it
    # would block a correct export over a defect that only affects solving.
    # Measured: ``Constraint(w, ">=", 5.0)`` emits ``r`` entry ``2 5.0``.
    model.validate(for_solve=False)
    text = _rust_nl_text(model)
    if text is None:
        writer = _NLWriter(model)
        text = writer.write()
    if path is not None:
        Path(path).write_text(text)
        return None
    return text


_RUST_NL_ENV = "DISCOPT_RUST_NL"

_RUST_NL_LOG = logging.getLogger(__name__)


def _rust_nl_text(model: Model) -> Optional[str]:
    """``.nl`` text from the Rust writer, or ``None`` to use the Python writer.

    Writing ``.nl`` was the entire remaining external-solver performance gap:
    the Python writer below costs **16.05 µs/row of a 16.21 µs/row**
    model-to-file pipeline, against oximo's 0.86 for the same work
    (``docs/dev/performance-plan.md`` §43). ``discopt_core::nl_writer`` is the
    replacement, and it is **9.3x faster** on a vectorised model (16.05 -> 1.72
    µs/row), which puts discopt at 1.66 µs/row end to end: **10.5x Pyomo and
    1.57x oximo**, measured in one run on one machine (§46).

    It is not a reimplementation with its own opinions: it is diffed
    **byte-for-byte** against the Python writer, and agrees on all 66 MINLPLib
    instances in ``python/tests/data/minlplib_nl`` plus a hand-written set
    covering MINLP integrality, matmul, vectorised bodies, equalities and
    free/fixed bounds. The Python writer stays as the fallback for anything the
    arena cannot represent (``dm.custom``), and ``DISCOPT_RUST_NL=0`` forces it,
    so a suspected regression can be isolated without a rebuild.
    """
    if os.environ.get(_RUST_NL_ENV, "1") in ("0", "false", "False"):
        return None
    try:
        from discopt._rust import model_to_repr
    except ImportError:  # pragma: no cover - extension always present in-tree
        return None
    # Builder-resident rows (`add_linear_constraints` / the `Model.constraint`
    # fast path) sit AHEAD of the expression rows in the arena and AFTER them in
    # `model._constraints`. That used to be a refusal here, which paired the
    # FASTEST construction path with the SLOWEST writer: a 20 000-row bulk model
    # built in 0.31 us/row and then exported at 22.83. `nl_writer::write_nl` now
    # takes the builder-row boundary from the repr and reorders to this order, so
    # the two writers agree byte for byte and row order -- how a solver's `.sol`
    # duals map back to constraints -- is unchanged from what discopt has always
    # written.
    try:
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        # Through a typed local: `write_nl` comes from the PyO3 extension, whose
        # bindings are untyped, so returning it directly returns `Any` from a
        # function declared to return `str | None`.
        text: str = repr_.write_nl(model.name)
        return text
    except Exception as exc:  # noqa: BLE001
        # "This model has no arena representation" arrives as several exception
        # types (`TypeError: Unknown expression type`, `ValueError: Unknown
        # MathFunc`, and the writer's own refusals), and the Python writer below
        # handles every one of them correctly. Nothing is suppressed: this is a
        # speculative fast path, and the model is written either way.
        _RUST_NL_LOG.debug("Rust .nl writer declined: %s: %s", type(exc).__name__, exc)
        return None
    except BaseException as exc:  # noqa: BLE001
        # PyO3 raises `pyo3_runtime.PanicException`, which derives from
        # `BaseException` and NOT from `Exception` -- so the clause above cannot
        # see it, and a Rust panic reached the user as a crash even though the
        # Python writer below handles the model correctly. That is exactly what
        # `log2` did: `expand.rs::func_code` admitted it, `nl_writer.rs` had no
        # opcode for it, and `.expect("mapped function")` fired.
        #
        # A panic is a BUG in the writer, not a decline, so unlike the clause
        # above this logs at WARNING: the file still gets written, but the defect
        # is not silent. KeyboardInterrupt and SystemExit are re-raised -- they
        # are control flow, never a writer result.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _RUST_NL_LOG.warning(
            "Rust .nl writer PANICKED (%s: %s); falling back to the Python writer. "
            "This is a writer bug -- please report it.",
            type(exc).__name__,
            exc,
        )
        return None


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


def _elem(arr: np.ndarray, idx: Any) -> Expression:
    """Read one element out of an ``object``-dtype array of expressions.

    ``ndarray`` is not generic over its ``object`` payload, so numpy's stubs
    type every element access as another array. These arrays only ever hold
    :class:`Expression` nodes; the cast records that once instead of silencing
    the checker at each of the five call sites.
    """
    return cast(Expression, arr[idx])


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
        # Variables referenced by each constraint's nonlinear body (excluding
        # pure constants). The Jacobian sparsity of a constraint is the UNION of
        # its linear vars and these; a nonlinear-only var gets a 0-coefficient
        # J entry (ASL convention). Filled by _decompose_expressions.
        self._con_nl_vars: list[set[int]] = []
        self._obj_nl_vars: set[int] = set()  # vars in the objective nonlinear body
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
        self._refuse_unrepresentable_relations()
        self._build_var_map()
        self._decompose_expressions()
        self._reorder_vars_canonical()
        self._compute_nl_vars()
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

    # ── Refuse relations the format cannot carry ──

    def _refuse_unrepresentable_relations(self):
        """Refuse a non-algebraic relation by name, before any row is written.

        Without this the disjunctive/SOS rows a GDP or an MPEC complementarity
        encoding leaves on the model reached ``_decompose_expressions`` and died
        on ``float(con.rhs)`` with ``AttributeError: '_DisjunctiveConstraint'
        object has no attribute 'rhs'`` (#1218) -- an internal error from deep
        inside the writer, where the real answer is that ``.nl`` has no such row
        and the model needs a further lowering first. Shared with the LP/MPS/GAMS
        writers, which walk ``model._constraints`` the same way.
        """
        refuse_non_algebraic_relations(self.model, ".nl")

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

    # ── Jacobian sparsity (union of linear + nonlinear vars per constraint) ──

    def _compute_nl_vars(self):
        """Resolve each constraint's nonlinear-body variable indices.

        Must run *after* :meth:`_reorder_vars_canonical` so ``_var_index`` is
        final (nonlinear expressions resolve their variable indices lazily
        through it). Stored in ``_con_nl_vars`` and used by the header, ``k``,
        and ``J`` writers so all three agree on one sparsity pattern.
        """
        self._con_nl_vars = []
        for nl in self._con_nonlinear:
            vs: set[int] = set()
            if nl is not None:
                self._collect_var_indices(nl, vs)
            self._con_nl_vars.append(vs)
        obj_vs: set[int] = set()
        if self._obj_nonlinear is not None:
            self._collect_var_indices(self._obj_nonlinear, obj_vs)
        self._obj_nl_vars = obj_vs

    def _jac_cols(self, i: int) -> dict[int, float]:
        """Full Jacobian row for constraint ``i``: ``{var_idx: coeff}``.

        The ASL convention: every variable appearing in the constraint —
        linearly *or* nonlinearly — gets an entry carrying its linear
        coefficient (``0.0`` for a variable that appears only nonlinearly).
        This single union map is the source of truth for the header nonzero
        count, the ``k`` cumulative-column-count section, and the ``J`` block,
        so all three are guaranteed to agree.
        """
        cols = dict(self._con_linear[i])
        for vi in self._con_nl_vars[i]:
            cols.setdefault(vi, 0.0)
        return cols

    def _obj_grad_cols(self) -> dict[int, float]:
        """Full objective gradient: ``{var_idx: coeff}`` (0.0 for nonlinear-only).

        The ``G`` section and the header gradient-nonzero count are both built
        from this union of the objective's linear vars and its nonlinear-body
        vars, so they agree.
        """
        cols = dict(self._obj_linear)
        for vi in self._obj_nl_vars:
            cols.setdefault(vi, 0.0)
        return cols

    # ── Decompose expressions into linear + nonlinear parts ──

    def _decompose_expressions(self):
        """Split objective and constraints into linear and nonlinear parts."""
        obj = self.model._objective
        if obj is not None:
            # The objective goes through the SAME array expansion as a constraint
            # body. It did not, and that was the single reason six of eight
            # array-shaped construct families could not be exported: an objective
            # like `-dm.sum(x)` or `dm.sum(A @ x)` is scalar-VALUED but
            # array-STRUCTURED, so it reached the scalar writer whole and died
            # with `Cannot write array variable x without indexing`. Array
            # *constraints* had worked all along.
            body = _arrays.scalarize_objective(obj.expression, self._resolve_var_index)
            # Objectives have no r-section rhs, so the constant offset stays in
            # the O body (re-attached as an additive node).
            linear, nonlinear, const_offset = self._split_expr(body)
            self._obj_linear = linear
            self._obj_nonlinear = self._attach_const(nonlinear, const_offset)
        self._decompose_builder_objective()

        for con in self.model._constraints:
            # Determine constraint bound type (shared by every scalar row this
            # constraint expands into).
            rhs = float(con.rhs)
            if con.sense == "<=":
                base_bnd = (1, 0.0, rhs)  # body <= rhs
            elif con.sense == ">=":
                base_bnd = (2, rhs, 0.0)  # body >= rhs
            elif con.sense == "==":
                base_bnd = (4, rhs, rhs)
            else:
                base_bnd = (3, 0.0, 0.0)  # free (unreachable in practice)

            # Vectorized constraints (e.g. DAE/collocation bodies built from
            # matrix products and broadcasting) carry an array-valued body that
            # encodes many scalar constraints at once. Expand it into one scalar
            # expression per output element before linear/nonlinear splitting.
            bodies = self._scalarize_body(con.body)
            for body in bodies:
                linear, nonlinear, const_offset = self._split_expr(body)
                self._con_linear.append(linear)
                self._con_nonlinear.append(nonlinear)
                self._con_bounds.append(self._offset_bound(base_bnd, const_offset))

        self._decompose_builder_blocks()

    @staticmethod
    def _attach_const(nl: Expression | None, const_offset: float) -> Expression | None:
        """Re-attach a constant offset to a nonlinear body (for objectives).

        Constraints carry the constant in the r-section rhs, but the objective
        has no rhs — so its constant offset is added back as an ``x + const``
        node (or a bare ``Constant`` when there is no variable part).
        """
        if const_offset == 0.0:
            return nl
        if nl is None:
            return Constant(const_offset)
        return BinaryOp("+", nl, Constant(const_offset))

    @staticmethod
    def _offset_bound(
        bnd: tuple[int, float, float], const_offset: float
    ) -> tuple[int, float, float]:
        """Move a body constant offset into the constraint bound.

        The split peels the constant out of the body, so ``body <sense> rhs``
        becomes ``(body - const) <sense> (rhs - const)``. Subtract the offset
        from whichever numeric bound(s) the sense uses.
        """
        if const_offset == 0.0:
            return bnd
        btype, lb, ub = bnd
        if btype == 1:  # <= ub
            return (btype, lb, ub - const_offset)
        if btype == 2:  # >= lb
            return (btype, lb - const_offset, ub)
        if btype in (0, 4):  # range / equality: both bounds carry rhs
            return (btype, lb - const_offset, ub - const_offset)
        return bnd  # free: nothing to shift

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
        for j, raw in enumerate(np.asarray(c, dtype=np.float64).ravel()):
            coeff = float(raw)
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
        """The scalar constraint bodies a (possibly array) body expands to."""
        return _arrays.scalarize_body(expr, self._resolve_var_index)

    def _needs_scalarize(self, expr: Expression) -> bool:
        """True if ``expr`` has array structure requiring element expansion."""
        return _arrays.needs_scalarize(expr, self._resolve_var_index)

    @staticmethod
    def _obj0(x) -> np.ndarray:
        return _arrays.obj0(x)

    def _sum_terms(self, terms: list[Expression]) -> Expression:
        return _arrays.sum_terms(terms)

    def _matmul_scalar(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return _arrays.matmul_scalar(left, right)

    def _sum_axis(self, arr: np.ndarray, axis: int | None) -> np.ndarray:
        return _arrays.sum_axis(arr, axis)

    def _scalarize_reduction(self, expr: FunctionCall) -> Expression:
        return _arrays.scalarize_reduction(expr)

    def _scalarize(self, expr: Expression) -> np.ndarray:
        return _arrays.scalarize(expr)

    def _split_expr(self, expr: Expression) -> tuple[dict[int, float], Expression | None, float]:
        """Split an expression into linear, variable-referencing nonlinear, and constant.

        Returns ``(linear, nonlinear, const_offset)`` where:

        - ``linear`` is ``{var_idx: coeff}`` for the linear terms,
        - ``nonlinear`` is the variable-referencing nonlinear remainder (or
          ``None`` if there is none), and
        - ``const_offset`` is the accumulated constant contribution.

        The constant is peeled out separately (rather than folded into the
        nonlinear body) so callers can carry it in the ``r``-section rhs
        instead of an ``n<const>`` node in the constraint body. Keeping the
        nonlinear body free of a pure constant is what lets a constraint like
        ``x + y <= 3`` (which discopt normalizes to ``x + y - 3 <= 0``) be
        emitted as a genuinely *linear* constraint with an ``n0`` body — the
        ASL convention (nonlinear constraints must be the first ``n_nl_cons``
        and carry the only non-``n0`` bodies).
        """
        linear: dict[int, float] = {}
        nonlinear_terms: list[Expression] = []
        const_acc: list[float] = [0.0]
        self._collect_linear(expr, 1.0, linear, nonlinear_terms, const_acc)
        const_offset = const_acc[0]
        if nonlinear_terms:
            if len(nonlinear_terms) == 1:
                nl_expr = nonlinear_terms[0]
            else:
                nl_expr = nonlinear_terms[0]
                for t in nonlinear_terms[1:]:
                    nl_expr = BinaryOp("+", nl_expr, t)
            return linear, nl_expr, const_offset
        return linear, None, const_offset

    def _collect_linear(
        self,
        expr: Expression,
        coeff: float,
        linear: dict[int, float],
        nonlinear: list[Expression],
        const_acc: list[float],
    ):
        """Collect linear terms ``{var_idx: coeff}`` and segregate nonlinear ones.

        Constant contributions accumulate into ``const_acc[0]`` (a 1-element
        list used as a mutable float cell) rather than the ``nonlinear`` list,
        so a pure constant never lands in the emitted nonlinear body — the
        caller folds it into the r-section rhs instead.

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
                    # A nonzero constant contributes a constant offset. Keep it
                    # out of the nonlinear body: callers fold it into the
                    # r-section rhs so a constant-only body stays ``n0``.
                    const_acc[0] += val * c
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

        # A constraint counts as nonlinear only when its nonlinear body
        # references at least one variable. Since _split_expr peels pure
        # constants out into the r-section rhs, a constant-only body leaves
        # _con_nonlinear[i] is None (and _con_nl_vars[i] empty), so it is
        # correctly counted as linear (ASL requires nonlinear constraints to be
        # the first n_nl_cons rows and carry the only non-n0 bodies).
        n_nl_cons = sum(1 for vs in self._con_nl_vars if vs)
        n_nl_objs = 1 if self._obj_nonlinear is not None else 0

        # Count Jacobian nonzeros from the SAME union sparsity the k and J
        # sections use: for each constraint, |union(linear vars, nonlinear
        # vars)|. A variable appearing both linearly and nonlinearly is counted
        # once; a nonlinear-only variable is counted (with a 0 coefficient in J).
        n_jac_nz = sum(len(self._jac_cols(i)) for i in range(len(self._con_linear)))

        # Objective gradient nonzeros: union of the objective's linear vars and
        # its nonlinear-body vars (same de-duplication).
        n_grad_nz = len(self._obj_grad_cols())

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
        """Cumulative per-column Jacobian nonzero counts.

        Built from the SAME union sparsity (:meth:`_jac_cols`) the ``J`` blocks
        and the header nonzero count use, so all three agree. A single pass over
        the constraint rows accumulates a per-column tally (fixes EX-9's former
        ``O(n_vars x n_cons)`` nested scan). ASL reads the ``k`` section as the
        cumulative count of nonzeros in columns ``0 .. n_vars-2``; the final
        (``n_vars-1``-th) column total is implied by the header nonzero count.
        """
        if self._n_total <= 1:
            buf.write("k0\n")
            return
        per_col = [0] * self._n_total
        for i in range(len(self._con_linear)):
            for col in self._jac_cols(i):
                per_col[col] += 1
        buf.write(f"k{self._n_total - 1}\n")
        cumulative = 0
        for col in range(self._n_total - 1):
            cumulative += per_col[col]
            buf.write(f"{cumulative}\n")

    # ── J sections (linear Jacobian terms) ──

    def _write_J_sections(self, buf: io.StringIO):
        """Per-constraint Jacobian blocks (ASL convention).

        Each block lists *every* variable appearing in the constraint —
        linearly or nonlinearly — with its linear coefficient (``0`` for a
        variable that appears only nonlinearly). This is the union sparsity the
        header and ``k`` section are computed from.
        """
        for i in range(len(self._con_linear)):
            cols = self._jac_cols(i)
            if not cols:
                continue
            buf.write(f"J{i} {len(cols)}\n")
            for var_idx in sorted(cols.keys()):
                buf.write(f"{var_idx} {cols[var_idx]}\n")

    # ── G section (linear objective gradient) ──

    def _write_G_section(self, buf: io.StringIO):
        cols = self._obj_grad_cols()
        if not cols:
            return
        buf.write(f"G0 {len(cols)}\n")
        for var_idx in sorted(cols.keys()):
            buf.write(f"{var_idx} {cols[var_idx]}\n")

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
            elif isinstance(node, Parameter):
                # A Parameter is "a value fixed during a single solve but
                # changeable between solves" (its class docstring), so at export
                # time it is simply a constant -- .nl has no parameter concept.
                # The written file is a SNAPSHOT at the current value; re-export
                # after changing `p.value` to get a file for the new value.
                # Refuse a non-scalar Parameter rather than writing its first
                # element: `_needs_scalarize` routes shaped bodies through
                # `_scalarize`, so a shaped Parameter reaching here means the
                # element was not resolved, and silently emitting one number
                # would produce a wrong model (CLAUDE.md §3).
                val = np.asarray(node.value)
                if val.shape not in ((), (1,)):
                    raise ValueError(
                        f"Cannot write array parameter {node.name!r} of shape "
                        f"{val.shape} to .nl without indexing; index it "
                        f"element-wise (e.g. p[i]) so each use is scalar."
                    )
                buf.write(f"n{float(val.reshape(-1)[0])}\n")
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
                    # An index into a Parameter is a constant at export time.
                    # Constraint bodies reach `_scalarize`, which expands shaped
                    # parameters element-wise, but the objective is written
                    # directly through this path, so `p[i]` in an objective must
                    # be resolved here too.
                    if isinstance(node.base, Parameter):
                        val = np.asarray(node.base.value)
                        try:
                            elem = val[node.index]
                        except (IndexError, TypeError) as exc:
                            raise ValueError(
                                f"Cannot resolve parameter index {node}: "
                                f"{node.base.name!r} has shape {val.shape}."
                            ) from exc
                        elem_arr = np.asarray(elem)
                        if elem_arr.ndim != 0:
                            raise ValueError(
                                f"Parameter index {node} selects a non-scalar of "
                                f"shape {elem_arr.shape}; index it element-wise so "
                                f"each use is scalar."
                            )
                        buf.write(f"n{float(elem_arr)}\n")
                        continue
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
