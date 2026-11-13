"""Shared model-row iteration for exporters, the classifier, and the examiner.

The fast-construction API (``Model.constraint(...)`` linear fast path and
``Model.add_linear_constraints``) emits rows directly into the Rust builder and
records them on ``model._builder_linear_blocks`` — they do **not** appear in
``model._constraints``. A consumer that reads only ``model._constraints`` therefore
sees a *subset* of the model, which for a classifier / extractor / exporter /
examiner is a silent-wrong hazard (false optimum, false certificate, empty export).
See ``docs/dev/review-execution-plan.md`` §1 (X-1).

This module provides the single shared view of *all* linear-form rows a model
carries — expression-path rows in ``model._constraints`` and builder-resident rows
in ``model._builder_linear_blocks`` — so every such consumer sees the whole model.
The builder-block decomposition mirrors ``export/nl.py:_decompose_builder_blocks``
(the one place that already handled these rows correctly for ``.nl`` export).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discopt.modeling.core import (
    Constraint,
    Model,
    Variable,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _LogicalConstraint,
    _SOSConstraint,
)


def has_builder_only_rows(model: Model) -> bool:
    """True if the model carries linear rows/objective that live only in the builder.

    These are the rows emitted by the fast-construction API
    (``add_linear_constraints`` / the ``Model.constraint`` linear fast path) and
    the ``add_linear_objective`` / ``add_quadratic_objective`` builder objective —
    none of which are mirrored into ``model._constraints`` / ``model._objective``.
    A ``True`` return means any consumer reading only ``_constraints`` /
    ``_objective`` sees a strict subset of the model.
    """
    if getattr(model, "_builder_linear_blocks", None):
        return True
    if getattr(model, "_builder_linear_objective", None) is not None:
        return True
    if getattr(model, "_builder_quadratic_objective", None) is not None:
        return True
    return False


def has_builder_only_constraint_rows(model: Model) -> bool:
    """True if the model carries constraint rows that live only in the builder.

    Narrower than :func:`has_builder_only_rows`: ignores a builder-resident
    *objective* (which the expression-path exporters recover separately). Used by
    consumers that only need to know whether constraint feasibility can be judged
    from ``model._constraints`` alone.
    """
    return bool(getattr(model, "_builder_linear_blocks", None))


@dataclass
class LinearRow:
    """A single linear constraint row in the export flat-variable coordinate system.

    ``coeffs`` maps flat variable index (the ordering produced by
    :func:`discopt.export._extract.flatten_variables`) to coefficient. ``sense`` is
    one of ``"<="``, ``">="``, ``"=="``. The row asserts
    ``sum(coeff * x[idx]) sense rhs``.
    """

    coeffs: dict[int, float]
    sense: str
    rhs: float
    name: str
    # (Variable, local_flat_index, coeff) triples — lets name-based writers (GAMS)
    # emit the row using variable names + element indices rather than flat indices.
    terms: list[tuple[Variable, int, float]] = field(default_factory=list)


def variable_flat_offsets(model: Model) -> dict[int, int]:
    """Map ``id(variable)`` -> its starting flat index in the export ordering.

    Matches :func:`discopt.export._extract.flatten_variables` /
    :func:`discopt.export._extract._var_index`: scalars occupy one column, arrays
    occupy ``size`` columns, in ``model._variables`` order.
    """
    offsets: dict[int, int] = {}
    offset = 0
    for v in model._variables:
        offsets[id(v)] = offset
        offset += max(v.size, 1) if v.shape != () else 1
    return offsets


def block_columns(x) -> tuple[list, np.ndarray, np.ndarray]:
    """The columns of a fast-API block's variable record ``x`` (a :class:`Variable` or
    a :class:`~discopt.modeling.core.VarBlock`): ``(variables, position, local)``,
    column ``k`` being flat element ``local[k]`` of ``variables[position[k]]``."""
    from discopt.modeling.core import VarBlock

    if isinstance(x, VarBlock):
        pos, local = x.columns()
        return list(x.vars), pos, local
    n = int(x.size) if x.shape != () else 1
    return [x], np.zeros(n, dtype=np.int64), np.arange(n)


def block_flat_columns(x, offsets: dict[int, int]) -> np.ndarray:
    """Flat model index of each column of block record ``x``, from ``offsets``
    (``id(variable)`` -> flat start, e.g. :func:`variable_flat_offsets`). Raises when a
    variable of the block is not in ``offsets``."""
    vs, pos, local = block_columns(x)
    starts = []
    for v in vs:
        base = offsets.get(id(v))
        if base is None:
            raise ValueError(
                f"Fast-API constraint block references variable '{getattr(v, 'name', v)}' "
                "that is not registered in the model; cannot export."
            )
        starts.append(base)
    return np.asarray(starts, dtype=np.int64)[pos] + local


def iter_builder_linear_rows(model: Model) -> list[LinearRow]:
    """Decompose ``model._builder_linear_blocks`` into per-row :class:`LinearRow`.

    Mirrors ``export/nl.py:_decompose_builder_blocks`` but resolves coefficients
    into the export flat-variable coordinate system (rather than the ``.nl`` global
    index) so MPS/LP/GAMS and the examiner see the same rows the solver does.
    """
    blocks = getattr(model, "_builder_linear_blocks", None)
    if not blocks:
        return []
    offsets = variable_flat_offsets(model)
    rows: list[LinearRow] = []
    for A, x, sense, b, name in blocks:
        flat = block_flat_columns(x, offsets)
        col_vars, col_pos, col_local = block_columns(x)
        indptr = A.indptr
        indices = A.indices
        data = A.data
        b_arr = np.broadcast_to(np.asarray(b, dtype=np.float64), (A.shape[0],))
        for r in range(A.shape[0]):
            coeffs: dict[int, float] = {}
            terms: list[tuple[Variable, int, float]] = []
            for k in range(int(indptr[r]), int(indptr[r + 1])):
                coeff = float(data[k])
                if coeff == 0.0:
                    continue
                col = int(indices[k])
                gidx = int(flat[col])
                coeffs[gidx] = coeffs.get(gidx, 0.0) + coeff
                terms.append((col_vars[int(col_pos[col])], int(col_local[col]), coeff))
            row_name = f"{name}_{r}" if name else f"blk{len(rows)}"
            rows.append(
                LinearRow(
                    coeffs=coeffs,
                    sense=sense,
                    rhs=float(b_arr[r]),
                    name=row_name,
                    terms=terms,
                )
            )
    return rows


def builder_objective(model: Model):
    """Recover a builder-resident objective as flat-index coefficients.

    ``add_linear_objective`` / ``add_quadratic_objective`` set the real objective in
    the Rust builder and leave a zero *placeholder* in ``model._objective``. When the
    current ``model._objective`` is that placeholder, this returns a
    ``(linear_coeffs, quad_coeffs, constant, sense)`` tuple in the export flat-variable
    coordinate system; otherwise (a real expression objective, or no builder
    objective) it returns ``None`` and the caller uses the expression objective.

    ``linear_coeffs`` maps flat index -> coefficient; ``quad_coeffs`` maps
    ``(i, j)`` with ``i <= j`` -> coefficient of ``x_i x_j`` in ``0.5 x'Qx`` (i.e.
    the same convention as ``export._extract.extract_quadratic_terms``), or ``None``
    for a purely linear objective. ``sense`` is ``"minimize"`` / ``"maximize"``.
    """
    obj = model._objective
    if obj is None or not getattr(obj, "_is_placeholder", False):
        return None
    lin_blk = getattr(model, "_builder_linear_objective", None)
    quad_blk = getattr(model, "_builder_quadratic_objective", None)
    offsets = variable_flat_offsets(model)

    def _coeffs_from_c(c, x) -> dict[int, float]:
        base = offsets.get(id(x))
        if base is None:
            raise ValueError(
                f"Builder objective references variable '{getattr(x, 'name', x)}' "
                "that is not registered in the model; cannot export."
            )
        out: dict[int, float] = {}
        for j, raw in enumerate(np.asarray(c, dtype=np.float64).ravel()):
            coeff = float(raw)
            if coeff != 0.0:
                out[base + j] = out.get(base + j, 0.0) + coeff
        return out

    if lin_blk is not None:
        c, x, constant, sense = lin_blk
        return _coeffs_from_c(c, x), None, float(constant), sense
    if quad_blk is not None:
        Q, c, x, constant, sense = quad_blk
        base = offsets.get(id(x))
        if base is None:
            raise ValueError(
                f"Builder objective references variable '{getattr(x, 'name', x)}' "
                "that is not registered in the model; cannot export."
            )
        import scipy.sparse as sp

        # Mirror the Rust builder: it optimises 0.5 x'Sx with S = triu(Q)+striu(Q).T.
        S = (sp.triu(Q, 0) + sp.triu(Q, 1).T).tocsr()
        quad: dict[tuple[int, int], float] = {}
        indptr, indices, data = S.indptr, S.indices, S.data
        for i in range(S.shape[0]):
            for k in range(int(indptr[i]), int(indptr[i + 1])):
                q = float(data[k])
                if q == 0.0:
                    continue
                j = int(indices[k])
                gi, gj = base + i, base + j
                key = (min(gi, gj), max(gi, gj))
                # extract_quadratic_terms convention stores the coefficient of x_i x_j
                # (not 0.5 Q): 0.5 S[i,i] x_i^2 -> 0.5*S on the diagonal key; for the
                # off-diagonal, S[i,j] and S[j,i] each contribute 0.5, summing to S[i,j].
                quad[key] = quad.get(key, 0.0) + (0.5 * q if gi == gj else 0.5 * q)
        return _coeffs_from_c(c, x), quad, float(constant), sense
    return None


def iter_all_rows(model: Model) -> tuple[list[Constraint], list[LinearRow]]:
    """Return (expression-path constraint rows, builder-resident linear rows).

    The two lists together are the *whole* constraint set of the model. Expression
    rows are returned as the original :class:`Constraint` objects (so DAG-walking
    consumers keep full fidelity); builder rows are returned as :class:`LinearRow`
    in the export flat-variable coordinate system. Consumers that must see the whole
    model (classifier, extractor, exporters, examiner) iterate both.
    """
    expr_rows = [c for c in model._constraints if isinstance(c, Constraint)]
    builder_rows = iter_builder_linear_rows(model)
    return expr_rows, builder_rows


# ── Relations no exporter here can carry (#1218) ─────────────────────────────
#
# Every writer in this package emits ordinary algebraic rows: a body, a sense and
# a right-hand side, plus variable bounds. A disjunction, an indicator, an SOS
# set and a propositional clause are *structure* -- none of them is such a row,
# and none of the writers emits the SOS/indicator sections some of these formats
# define. Each of them nevertheless walked ``model._constraints`` assuming
# ``Constraint``, so a GDP or MPEC model died inside the writer on
# ``AttributeError: '_DisjunctiveConstraint' object has no attribute 'rhs'``
# (``.body`` for LP/MPS/GAMS) -- an internal error where the real answer is that
# the model needs a further lowering first. The table maps each IR type to how it
# reads in the model and to that lowering, so the refusal names both.
_GDP_REMEDY = (
    "Lower it to algebra first and export the lowered model:\n"
    "    from discopt._relax.gdp_reformulate import reformulate_gdp\n"
    "    lowered = reformulate_gdp(model, method='big-m')   # or method='hull'\n"
    "(the lowering adds selector binaries and needs finite bounds on the "
    "variables of each disjunct)."
)

_UNREPRESENTABLE_RELATIONS: dict[type, tuple[str, str]] = {
    _DisjunctiveConstraint: (
        "a disjunctive constraint (Model.either_or / Model.if_else, GDP)",
        _GDP_REMEDY,
    ),
    _IndicatorConstraint: (
        "an indicator constraint (Model.if_then)",
        _GDP_REMEDY,
    ),
    _SOSConstraint: (
        "an SOS constraint (Model.sos1 / Model.sos2)",
        "No SOS section is emitted by this writer. " + _GDP_REMEDY,
    ),
    _LogicalConstraint: (
        "a propositional-logic constraint over BooleanVars",
        _GDP_REMEDY,
    ),
}


def refuse_non_algebraic_relations(model: Model, fmt: str) -> None:
    """Refuse a relation the writer cannot carry, by name, before writing a row.

    Parameters
    ----------
    model : Model
        The model about to be exported.
    fmt : str
        The target format, as it appears at the head of the error message
        (``".nl"``, ``"LP"``, ``"MPS"``, ``"GAMS"``).

    Raises
    ------
    ValueError
        On the first non-``Constraint`` relation in ``model._constraints``,
        naming what it is and the lowering that makes the model exportable.
    """
    for con in model._constraints:
        if isinstance(con, Constraint):
            continue
        what, remedy = _UNREPRESENTABLE_RELATIONS.get(
            type(con),
            (
                f"a non-algebraic relation ({type(con).__name__})",
                "Reformulate it into ordinary algebraic constraints before exporting.",
            ),
        )
        name = getattr(con, "name", None)
        where = f" named {name!r}" if name else ""
        raise ValueError(
            f"{fmt} export: the model carries {what}{where}. This writer emits "
            "ordinary algebraic rows and variable bounds only, so there is "
            f"nothing faithful to write. {remedy}"
        )


def binary_box_is_default(lb, ub) -> bool:
    """True if a binary variable still carries its full declared ``[0, 1]`` box.

    Every writer has a shorthand for a binary column — MPS ``BV``, an LP
    ``Binaries`` listing, a GAMS ``Binary Variables`` declaration — and each
    shorthand *means* ``0 <= x <= 1``. Emitting it for a binary whose box has
    been narrowed (``x.fix(1)``, ``x.fix(1, where=mask)``, or a presolve-style
    pin) silently widens the column back out, so the exported file is a
    **relaxation** of the model discopt was handed and an external solver
    answers a different question. Measured before this guard existed: a model
    with ``x`` pinned at 1 and objective ``x + y`` exported to MPS and to LP,
    read back with HiGHS, optimized to **0** instead of 1 (EX-3).

    Writers therefore take the shorthand only when this returns True and fall
    through to their ordinary bound-emitting path otherwise. The test is
    elementwise so a partially pinned array (``where=`` mask) is caught: one
    narrowed element is enough to disqualify the whole block, since the
    shorthand is declared per variable, not per element.
    """
    lo = np.asarray(lb, dtype=np.float64)
    hi = np.asarray(ub, dtype=np.float64)
    return bool(np.all(lo <= 0.0) and np.all(hi >= 1.0))


# ── File-format-legal, collision-free names (#1613 C-03a) ──
#
# MPS and LP are whitespace-tokenised formats: a name with a space in it is two
# tokens, so a reader sees a *different model* rather than an error. Measured on
# the issue witness: free MPS with column ``x[LSR naphtha,regular]`` read by
# HiGHS optimised to 8.0 where the model's optimum is 24.0. Sanitising alone is
# not enough either: two distinct names can sanitise to the same token (``"a b"``
# and ``"a_b"``), and the flattening of an array ``x`` (``x_0``, ``x_1``) can
# already collide with a scalar named ``x_0``. A shared token is a *merged*
# column or row, i.e. another silently different model. So every exporter that
# writes names goes through :func:`legal_unique_names`, which both maps each
# name into the format's legal alphabet and de-duplicates the results.

# CPLEX LP name alphabet: letters, digits and these punctuation characters --
# minus ``/``, which HiGHS's LP reader rejects inside a name (measured).
_LP_NAME_PUNCT = frozenset("!\"#$%&(),.;?@_`'{}|~")
# Words an LP reader treats as section/bound keywords (case-insensitive). A
# column or row spelled like one is mis-parsed, so it gets a suffix.
_LP_KEYWORDS = frozenset(
    {
        "minimize", "minimise", "minimum", "min", "maximize", "maximise", "maximum",
        "max", "subject", "to", "st", "s.t.", "such", "that", "bounds", "bound",
        "free", "inf", "infinity", "general", "generals", "gen", "integer",
        "integers", "binary", "binaries", "bin", "semi", "semis",
        "semi-continuous", "sos", "end",
    }
)  # fmt: skip


def _legal_mps_name(name: str) -> str:
    # Free MPS separates fields by whitespace; anything that is not a printable,
    # non-space ASCII character is replaced. ``-`` is also replaced, matching the
    # row-name sanitiser this replaces.
    out = "".join(ch if (33 <= ord(ch) <= 126 and ch != "-") else "_" for ch in name)
    return out or "_"


def _legal_lp_name(name: str) -> str:
    out = "".join(
        ch if (ch.isascii() and ch.isalnum()) or ch in _LP_NAME_PUNCT else "_" for ch in name
    )
    if not out:
        return "_"
    # A name may not start with a digit or a period (it would read as a number),
    # nor with ``inf``/``nan`` in any case: HiGHS reads ``inf_``/``info``/
    # ``nanx`` as a number and rejects the file (measured).
    if out[0].isdigit() or out[0] == "." or out[:3].lower() in ("inf", "nan"):
        out = "_" + out
    if out.lower() in _LP_KEYWORDS:
        out = out + "_"
    return out


def legal_unique_names(names: list[str], fmt: str, reserved: tuple[str, ...] = ()) -> list[str]:
    """Map *names* to tokens legal in *fmt* (``"mps"`` or ``"lp"``), all distinct.

    A name that is already legal and unused is returned unchanged, so a model
    with clean names exports byte-identically. A name that is illegal is mapped
    into the format's alphabet; a result that collides with an earlier one (or
    with a *reserved* token such as the objective row's name) gets the first
    free ``_<k>`` suffix. Distinct inputs therefore always give distinct
    outputs — a merge of two columns or rows is impossible by construction.
    """
    if fmt == "mps":
        legal = _legal_mps_name
    elif fmt == "lp":
        legal = _legal_lp_name
    else:
        raise ValueError(f"unknown name format {fmt!r}")
    used: set[str] = set(reserved)
    out: list[str] = []
    for raw in names:
        cand = legal(raw)
        if cand in used:
            k = 1
            while f"{cand}_{k}" in used:
                k += 1
            cand = f"{cand}_{k}"
        used.add(cand)
        out.append(cand)
    return out
