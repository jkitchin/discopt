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
        base = offsets.get(id(x))
        if base is None:
            raise ValueError(
                f"Fast-API constraint block references variable '{getattr(x, 'name', x)}' "
                "that is not registered in the model; cannot export."
            )
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
                local = int(indices[k])
                gidx = base + local
                coeffs[gidx] = coeffs.get(gidx, 0.0) + coeff
                terms.append((x, local, coeff))
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
