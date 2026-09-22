"""Resolve declared block structure into the emitted NLP's index space (#1370).

POUNCE can factorize a block-structured KKT system block-parallel over a shared
border, and it takes the partition as **model-space labels**: one integer per
variable and one per constraint, negative for the shared ones
(``Problem.set_block_structure``). discopt knows those labels — the model
declares them through :meth:`~discopt.modeling.core.Model.set_block` /
:meth:`~discopt.modeling.core.Model.set_constraint_block` — and this module is
what turns a declaration into the two integer sequences the solver reads.

**The work here is not the format, it is stable identity.** A label is only
meaningful against the index space the NLP is actually emitted in, and a label
that is merely *in range* after a column moves names a neighbouring variable's
block — plausible, wrong, and undetectable downstream. POUNCE validates what it
can (a wrong length, an entry coupling two blocks, an oversized border) and falls
back with a warning, but it cannot detect a permutation that is still internally
consistent. So two rules hold here:

1. **The labels are assembled from the evaluator's own maps**, never from
   re-derived offset arithmetic: variable offsets come from ``model._variables``
   sizes cross-checked against ``evaluator.n_variables``, and rows come from
   ``evaluator.constraint_row_map()`` — which is *not* the identity, since an
   array-valued body is one ``Constraint`` and many rows, and the fast-builder
   rows are appended after ``model._constraints`` (#908, #840).
2. **The result is cross-checked against the emitted problem's own sparsity.**
   Every constraint row must lie inside one block plus the border, and no
   Lagrangian-Hessian entry may couple two blocks — POUNCE's arrowhead is over
   the whole KKT matrix, not just the Jacobian. A permutation error almost always
   violates one of the two, which is what makes an off-by-one *fail* rather than
   silently mean something else.

Both checks report an **executed-comparison count** (:class:`BlockChecks`), and
a caller that asserts a partition was validated should assert those counts are
non-zero: a validator that traverses nothing otherwise reads exactly like a
validator that found nothing wrong (CLAUDE.md, measurement discipline §6).

An inconsistent declaration is refused, loudly, rather than downgraded to "no
structure": it means the model says something about itself that is not true, and
that is a modelling bug worth a traceback, not a silent fall back to the
full-space solve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from discopt.modeling.core import Model

_logger = logging.getLogger(__name__)


class BlockStructureError(ValueError):
    """A declared block structure is inconsistent with the emitted NLP."""


@dataclass(frozen=True)
class BlockChecks:
    """How much validation actually ran (CLAUDE.md measurement discipline §6)."""

    rows_checked: int = 0
    jacobian_entries_checked: int = 0
    hessian_entries_checked: int = 0

    @property
    def total(self) -> int:
        return self.rows_checked + self.jacobian_entries_checked + self.hessian_entries_checked


@dataclass(frozen=True)
class BlockStructure:
    """A validated partition in the emitted NLP's index space.

    ``var_blocks[i]`` is the block of NLP column ``i`` and ``con_blocks[j]`` the
    block of NLP row ``j``; both are ``-1`` for the shared/linking border. Block
    ids are dense (``0 .. n_blocks-1``, ascending in declared-id order), so every
    block is non-empty by construction.
    """

    var_blocks: np.ndarray
    con_blocks: np.ndarray
    n_blocks: int
    border_dim: int
    linking_rows: int
    checks: BlockChecks

    def as_pair(self) -> tuple[list[int], list[int]]:
        """``(var_blocks, con_blocks)`` as plain Python ints, for the PyO3 boundary."""
        return [int(b) for b in self.var_blocks], [int(b) for b in self.con_blocks]

    def summary(self) -> str:
        return (
            f"BlockStructure: {self.n_blocks} blocks, border {self.border_dim} of "
            f"{self.var_blocks.size} columns, {self.linking_rows} linking of "
            f"{self.con_blocks.size} rows"
        )


def has_declaration(model: "Model") -> bool:
    """Does *model* declare a block partition at all?

    Whole-variable ``set_block`` annotations count: they are the same
    declaration at coarser granularity. ``set_stage`` (Benders staging) does
    not — a stage is an ordering, not a partition.
    """
    return bool(getattr(model, "_block_labels_var", None) or getattr(model, "_decomp_blocks", None))


def _declared_variable_labels(model: "Model", n_expected: int) -> np.ndarray:
    """One raw (un-densified) block id per NLP column; ``-1`` where undeclared.

    Element-wise labels (``set_block(V, ids)``) win over the whole-variable
    annotation for the same name; ``set_block`` keeps the two stores disjoint.
    """
    elementwise: dict = getattr(model, "_block_labels_var", {}) or {}
    whole: dict = getattr(model, "_decomp_blocks", {}) or {}

    labels: list[np.ndarray] = []
    total = 0
    for var in model._variables:
        size = int(var.size) if var.shape != () else 1
        total += size
        if var.name in elementwise:
            declared = np.asarray(elementwise[var.name], dtype=np.int64).reshape(-1)
            if declared.size != size:
                raise BlockStructureError(
                    f"variable {var.name!r} carries {declared.size} element block labels but "
                    f"occupies {size} NLP columns. Re-declare it with "
                    f"Model.set_block({var.name!r}, ids) against its current shape."
                )
            labels.append(declared)
        elif var.name in whole:
            labels.append(np.full(size, int(whole[var.name]), dtype=np.int64))
        else:
            labels.append(np.full(size, -1, dtype=np.int64))

    if total != n_expected:
        raise BlockStructureError(
            f"the model's flat variable width is {total} but the emitted NLP has "
            f"{n_expected} columns. Block labels are only meaningful against the index "
            "space the NLP is actually emitted in, so this partition is refused rather "
            "than shifted onto the wrong columns."
        )
    if not labels:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(labels)


def _declared_row_labels(model: "Model", evaluator: Any, m_expected: int) -> np.ndarray:
    """Explicitly declared row labels; ``-2`` marks "not declared, derive it"."""
    store: dict = getattr(model, "_block_labels_con", {}) or {}
    out = np.full(m_expected, -2, dtype=np.int64)
    if not store:
        return out

    matched: set = set()
    row_names: list[str] = []
    for start, stop, con in evaluator.constraint_row_map():
        cname = getattr(con, "name", None)
        if cname:
            row_names.append(cname)
        declared = store.get(id(con))
        if declared is not None:
            matched.add(id(con))
            if cname and cname in store:
                matched.add(cname)
        elif cname:
            declared = store.get(cname)
            if declared is not None:
                matched.add(cname)
        if declared is None:
            continue
        declared = np.asarray(declared, dtype=np.int64).reshape(-1)
        width = stop - start
        if declared.size == 1:
            out[start:stop] = int(declared[0])
        elif declared.size == width:
            out[start:stop] = declared
        else:
            name = getattr(con, "name", None) or f"rows {start}:{stop}"
            raise BlockStructureError(
                f"constraint {name!r} carries {declared.size} row block labels but occupies "
                f"{width} NLP rows. An array-valued body is one Constraint and many rows; "
                "declare one id, or exactly one per row."
            )

    unmatched = [k for k in store if k not in matched]
    if unmatched:
        # A declaration that matches no row is the silent-miss case: the label
        # simply never applies, and the partition POUNCE gets is not the one the
        # model believes it declared. Two ways to land here, both worth a
        # traceback: a stale object handle (the fast-builder rows are
        # materialized fresh on every read, so their `id` never matches), and a
        # name that is not a row name (`add_linear_constraints(name="border")`
        # produces rows named `border_0`, `border_1`, ...).
        shown = ", ".join(repr(k) if isinstance(k, str) else f"<object {k}>" for k in unmatched)
        raise BlockStructureError(
            f"constraint block labels were declared for {shown}, which matches no row of the "
            f"emitted NLP. Row names available: {sorted(set(row_names)) or '(none named)'}. "
            "Declare fast-builder rows by their per-row names."
        )
    return out


def resolve_block_structure(
    model: "Model",
    evaluator: Any,
    *,
    check_hessian: bool = True,
) -> BlockStructure:
    """Resolve *model*'s declaration against *evaluator*'s emitted NLP.

    Args:
        model: the declaring model (``set_block`` / ``set_constraint_block``).

        evaluator: the NLP evaluator that will serve the solve — the authority on
            the index space. Its ``constraint_row_map``, ``jacobian_structure``
            and ``hessian_structure`` are what the labels are checked against.

        check_hessian: run the Lagrangian-Hessian cross-block check. On by
            default; it is the stronger of the two off-by-one detectors, because
            it sees couplings the Jacobian does not (an objective term over two
            blocks' variables). Only turn it off for an evaluator that cannot
            report a Hessian structure.

    Returns:
        A validated :class:`BlockStructure`.

    Raises:
        BlockStructureError: the declaration and the emitted NLP disagree.
    """
    n = int(evaluator.n_variables)
    m = int(evaluator.n_constraints)
    raw_var = _declared_variable_labels(model, n)
    explicit = _declared_row_labels(model, evaluator, m)
    return _build(raw_var, explicit, evaluator, check_hessian=check_hessian)


def validate_block_labels(
    labels: tuple[Any, Any],
    evaluator: Any,
    *,
    check_hessian: bool = True,
) -> BlockStructure:
    """Validate an explicitly supplied ``(var_blocks, con_blocks)`` pair.

    Same checks a resolved declaration gets — hand-built labels are exactly the
    ones most likely to be off by one, so they are not trusted more than a
    declaration is.
    """
    var_blocks, con_blocks = labels
    raw_var = np.asarray(var_blocks, dtype=np.int64).reshape(-1)
    raw_con = np.asarray(con_blocks, dtype=np.int64).reshape(-1)
    n = int(evaluator.n_variables)
    m = int(evaluator.n_constraints)
    if raw_var.size != n or raw_con.size != m:
        raise BlockStructureError(
            f"block labels have {raw_var.size} variable and {raw_con.size} constraint "
            f"entries, but the emitted NLP has {n} columns and {m} rows."
        )
    return _build(raw_var, raw_con, evaluator, check_hessian=check_hessian)


def _build(
    raw_var: np.ndarray,
    explicit: np.ndarray,
    evaluator: Any,
    *,
    check_hessian: bool,
) -> BlockStructure:
    """Densify, derive undeclared rows, and cross-check against the NLP's sparsity.

    ``explicit[j] == -2`` means "no row declaration, derive it from the columns
    the row touches".
    """
    m = int(evaluator.n_constraints)
    declared_ids = sorted({int(b) for b in raw_var if b >= 0})
    if len(declared_ids) < 2:
        raise BlockStructureError(
            f"a block partition needs at least two blocks; this declaration has "
            f"{len(declared_ids)}. Assign the block variables with Model.set_block(var, k) "
            "and the shared ones with a negative id."
        )
    dense = {old: new for new, old in enumerate(declared_ids)}
    var_blocks = np.array([dense[int(b)] if b >= 0 else -1 for b in raw_var], dtype=np.int64)

    jac_rows, jac_cols = evaluator.jacobian_structure()
    jac_rows = np.asarray(jac_rows, dtype=np.int64)
    jac_cols = np.asarray(jac_cols, dtype=np.int64)
    jac_entries = int(jac_rows.size)

    # Per-row column membership in one vectorised pass: a row's non-shared
    # columns all agree (min == max) or the row spans blocks and is linking.
    row_min, row_max = _row_block_extent(jac_rows, jac_cols, var_blocks, m)
    row_block = np.where(row_max < 0, -1, np.where(row_min == row_max, row_max, -1))
    row_spans = (row_max >= 0) & (row_min != row_max)

    undeclared = explicit == -2
    con_blocks = np.where(undeclared, row_block, np.where(explicit < 0, -1, explicit))
    rows_checked = int(explicit.size)

    stated = ~undeclared & (explicit >= 0)
    if stated.any():
        unknown = stated & ~np.isin(explicit, np.array(declared_ids, dtype=np.int64))
        if unknown.any():
            j = int(np.flatnonzero(unknown)[0])
            raise BlockStructureError(
                f"NLP row {j} is declared in block {int(explicit[j])}, which no variable is "
                f"declared in. Declared blocks: {declared_ids}."
            )
        mapped = np.array([dense.get(int(b), -1) for b in explicit], dtype=np.int64)
        con_blocks = np.where(stated, mapped, con_blocks)

        clash = stated & (row_block >= 0) & (mapped != row_block)
        if clash.any():
            j = int(np.flatnonzero(clash)[0])
            raise BlockStructureError(
                f"NLP row {j} is declared in block {int(explicit[j])} but its columns lie in "
                f"block {declared_ids[int(row_block[j])]}. A row's dual must be placed with "
                "the variables it actually touches, or the eliminated block is singular."
            )
        empty = stated & (row_max < 0)
        if empty.any():
            j = int(np.flatnonzero(empty)[0])
            raise BlockStructureError(
                f"NLP row {j} is declared in block {int(explicit[j])} but touches no column of "
                "that block — only shared ones. Its dual would be eliminated with a block that "
                "has no entry for it, leaving that block singular; such a row belongs on the "
                "border (a negative id)."
            )
        spanning = stated & row_spans
        if spanning.any():
            j = int(np.flatnonzero(spanning)[0])
            raise BlockStructureError(
                f"NLP row {j} is declared in block {int(explicit[j])} but spans more than one "
                "block. Declare it linking (a negative id), or move the variables it couples."
            )

    hess_entries = 0
    if check_hessian:
        hess_entries = _check_hessian(evaluator, var_blocks, declared_ids)

    border = int(np.count_nonzero(var_blocks < 0))
    linking = int(np.count_nonzero(con_blocks < 0))
    return BlockStructure(
        var_blocks=var_blocks,
        con_blocks=con_blocks,
        n_blocks=len(declared_ids),
        border_dim=border,
        linking_rows=linking,
        checks=BlockChecks(
            rows_checked=rows_checked,
            jacobian_entries_checked=jac_entries,
            hessian_entries_checked=hess_entries,
        ),
    )


def _row_block_extent(
    jac_rows: np.ndarray, jac_cols: np.ndarray, var_blocks: np.ndarray, m: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row ``(min, max)`` block id over the row's NON-shared columns.

    ``(-1, -1)`` for a row with no non-shared column at all — a row over only
    border variables, which belongs on the border. Grouped with a stable sort
    and ``reduceat`` rather than ``ufunc.at``: this runs once per solve over the
    whole Jacobian pattern, which on the models this feature exists for is
    millions of entries.
    """
    row_min = np.full(m, -1, dtype=np.int64)
    row_max = np.full(m, -1, dtype=np.int64)
    if jac_rows.size == 0:
        return row_min, row_max
    col_block = var_blocks[jac_cols]
    keep = col_block >= 0
    rows_k = jac_rows[keep]
    if rows_k.size == 0:
        return row_min, row_max
    blocks_k = col_block[keep]
    order = np.argsort(rows_k, kind="stable")
    rs = rows_k[order]
    bs = blocks_k[order]
    starts = np.flatnonzero(np.concatenate(([True], rs[1:] != rs[:-1])))
    group_rows = rs[starts]
    row_min[group_rows] = np.minimum.reduceat(bs, starts)
    row_max[group_rows] = np.maximum.reduceat(bs, starts)
    return row_min, row_max


def _check_hessian(evaluator: Any, var_blocks: np.ndarray, declared_ids: list[int]) -> int:
    """Refuse a Lagrangian-Hessian entry coupling two blocks; return entries checked.

    The arrowhead POUNCE solves is the whole KKT matrix, so a second-derivative
    term over two blocks' variables breaks the partition just as a constraint row
    would — and the Jacobian never sees it when it comes from the objective.
    """
    try:
        rows, cols = evaluator.hessian_structure()
    except (AttributeError, NotImplementedError):  # pragma: no cover - evaluator-dependent
        return 0
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.size == 0:
        return 0
    a = var_blocks[rows]
    b = var_blocks[cols]
    bad = (a >= 0) & (b >= 0) & (a != b)
    if bad.any():
        k = int(np.flatnonzero(bad)[0])
        i, j = int(rows[k]), int(cols[k])
        raise BlockStructureError(
            f"the Lagrangian Hessian couples column {i} (block {declared_ids[int(a[k])]}) with "
            f"column {j} (block {declared_ids[int(b[k])]}). Blocks may meet only through the "
            "shared border, so this partition would not be the arrowhead it claims to be. "
            "Declare the coupling variables shared (a negative block id)."
        )
    return int(rows.size)


def block_structure_for_model(
    model: "Model", evaluator: Any, *, required: bool = False
) -> Optional[BlockStructure]:
    """Resolve *model*'s declaration, or ``None`` when it declares nothing.

    ``required=True`` turns "nothing declared" into a refusal, for a caller that
    asked for the block path explicitly and would otherwise get a silent
    full-space solve.

    **Two kinds of declaration, and they are not refused alike.** The element-wise
    API (:meth:`Model.set_block` with an array,
    :meth:`Model.set_constraint_block`) exists only for this feature, so a
    partition it declares that is not an arrowhead is a modelling bug and
    raises. Whole-variable ``set_block``/``first_stage`` annotations predate it
    and mean "this is how the model decomposes for Benders/Lagrangian" — a
    partition perfectly valid there can be a poor arrowhead here (its blocks may
    meet in the objective), and failing an ordinary solve over it would be this
    feature punishing a model for an annotation aimed at another one. Those are
    logged and solved full-space instead.
    """
    if not has_declaration(model):
        if required:
            raise BlockStructureError(
                "this model declares no block structure. Use Model.set_block(var, k) to "
                "assign variables to blocks and a negative id for the shared ones."
            )
        return None
    declared_for_kkt = bool(
        getattr(model, "_block_labels_var", None) or getattr(model, "_block_labels_con", None)
    )
    try:
        return resolve_block_structure(model, evaluator)
    except BlockStructureError:
        if declared_for_kkt or required:
            raise
        _logger.info(
            "the model's decomposition annotations do not form a block-structured KKT "
            "partition; solving full-space. Declare one explicitly with "
            "Model.set_block(var, ids) to see why [block-structure-declined].",
            exc_info=True,
        )
        return None
