"""Element-level incidence graph and index coarsening.

:class:`~discopt.decomposition.graph.base.ModelGraph` is built from
``_collect_variables``, which maps every ``x[s, t]`` back to the whole array
``x``. Its vertices are therefore *arrays*, not scalars: a model whose
scenarios share one array variable reads as a single block even when fixing a
few first-stage variables would split it per scenario. This module builds the
same bipartite variable–constraint graph one level down, with one vertex per
scalar column and one per scalar constraint row, and each vertex tagged with
the array it came from and its index tuple inside it.

Those tags are what make *index coarsening* possible (Allen & Westerberg 1976;
Kalamaris & Maravelias 2026, §3.1): merging the vertices of an array that
differ only along chosen axes gives a reduced graph in which repeated structure
(time steps, generators) collapses while the axes that carry the decomposition
(scenarios, periods) stay visible. :meth:`ElementGraph.coarsen` does that, and
:meth:`CoarseGraph.lift` maps a partition of the reduced graph back to columns.

**Source of truth.** Rows and columns are taken from the evaluator's own maps,
as :mod:`discopt.block_structure` does and for the same reason: an array-valued
body is one ``Constraint`` and many rows, and re-deriving that layout here
would be a second implementation that can drift from the one the solver uses.
The incidence is the evaluator's Jacobian sparsity pattern. A pattern entry
that is structurally present but numerically zero can only *merge* blocks,
never separate them, so the graph errs toward fewer blocks, which is the safe
direction for anything that later treats a split as exact.

**Scope.** Like the array-level layer, the graph is built from constraint rows
only; the objective is assumed separable across blocks (the standard
Lagrangian setup) and is not an edge source here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from discopt.decomposition.graph import kernels


class ElementGraphError(ValueError):
    """The evaluator's row/column layout does not match the model's."""


def _shape_of(obj: Any, size: int) -> tuple[int, ...]:
    """The index shape of a variable or constraint body occupying *size* slots.

    Falls back to ``(size,)`` when the object carries no shape whose product is
    *size*; the index tuple is then the row offset, which is still a unique,
    stable label.
    """
    shape = getattr(obj, "shape", None)
    if shape is None:
        return (size,)
    shape = tuple(int(d) for d in shape)
    if shape == ():
        return () if size == 1 else (size,)
    if int(np.prod(shape)) != size:
        return (size,)
    return shape


def _unravel(k: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if shape == ():
        return ()
    return tuple(int(i) for i in np.unravel_index(k, shape))


@dataclass(frozen=True)
class ElementChecks:
    """Executed-comparison counts for the build (CLAUDE.md measurement §6).

    A caller asserting the graph was cross-checked should assert these are
    non-zero: a build that walked nothing reads exactly like an empty model.
    """

    columns_labelled: int
    rows_labelled: int
    jacobian_entries: int


@dataclass(frozen=True)
class ElementGraph:
    """Scalar-level bipartite variable–constraint graph of a model.

    Attributes
    ----------
    var_names, var_shapes : tuple
        One entry per ``model._variables`` item, in declared order.
    col_var : np.ndarray
        ``col_var[j]`` is the position in ``var_names`` of flat column ``j``.
    col_index : tuple[tuple[int, ...], ...]
        ``col_index[j]`` is column ``j``'s index tuple inside its array (C order,
        the layout ``_flat_index`` and the evaluators share).
    con_names, con_shapes : tuple
        One entry per source constraint of the evaluator's row map, in row order.
        ``con_names[k]`` is the constraint's name, or ``"c{k}"`` when unnamed.
    row_con : np.ndarray
        ``row_con[i]`` is the position in ``con_names`` of row ``i``.
    row_index : tuple[tuple[int, ...], ...]
        Row ``i``'s index tuple inside its constraint body.
    row_cliques : tuple[np.ndarray, ...]
        ``row_cliques[i]`` is the sorted array of columns row ``i`` touches.
    checks : ElementChecks
    """

    var_names: tuple[str, ...]
    var_shapes: tuple[tuple[int, ...], ...]
    col_var: np.ndarray
    col_index: tuple[tuple[int, ...], ...]
    con_names: tuple[str, ...]
    con_shapes: tuple[tuple[int, ...], ...]
    row_con: np.ndarray
    row_index: tuple[tuple[int, ...], ...]
    row_cliques: tuple[np.ndarray, ...]
    checks: ElementChecks

    # ── construction ────────────────────────────────────────────

    @classmethod
    def from_model(cls, model, evaluator: Any = None) -> "ElementGraph":
        """Build the graph from *model* (and its evaluator, built if not given).

        Raises :class:`ElementGraphError` when the evaluator's column count or
        row map disagrees with the model; a label that is merely in range after
        a shift names a neighbouring element, so a mismatch is refused rather
        than guessed around.
        """
        if evaluator is None:
            from discopt._tape_nlp_evaluator import make_evaluator

            evaluator = make_evaluator(model)

        # Columns: declared order, C-order inside each array.
        var_names: list[str] = []
        var_shapes: list[tuple[int, ...]] = []
        col_var_parts: list[np.ndarray] = []
        col_index: list[tuple[int, ...]] = []
        for p, v in enumerate(model._variables):
            size = int(v.size) if v.shape != () else 1
            shape = _shape_of(v, size)
            var_names.append(v.name)
            var_shapes.append(shape)
            col_var_parts.append(np.full(size, p, dtype=np.int64))
            col_index.extend(_unravel(k, shape) for k in range(size))
        n = len(col_index)
        if n != int(evaluator.n_variables):
            raise ElementGraphError(
                f"the model's flat variable width is {n} but the evaluator has "
                f"{int(evaluator.n_variables)} columns; element labels would be shifted."
            )
        col_var = np.concatenate(col_var_parts) if col_var_parts else np.zeros(0, np.int64)

        # Rows: the evaluator's own row map, which must tile [0, m) exactly.
        m = int(evaluator.n_constraints)
        con_names: list[str] = []
        con_shapes: list[tuple[int, ...]] = []
        row_con = np.full(m, -1, dtype=np.int64)
        row_index: list[tuple[int, ...]] = [()] * m
        expected = 0
        for k, (start, stop, con) in enumerate(evaluator.constraint_row_map()):
            start, stop = int(start), int(stop)
            if start != expected or stop < start or stop > m:
                raise ElementGraphError(
                    f"the evaluator's row map is not a tiling of its {m} rows "
                    f"(entry {k} covers [{start}, {stop}), expected to start at {expected})."
                )
            width = stop - start
            body = getattr(con, "body", None)
            shape = _shape_of(body, width)
            name = getattr(con, "name", None)
            con_names.append(str(name) if name else f"c{k}")
            con_shapes.append(shape)
            row_con[start:stop] = k
            for r in range(width):
                row_index[start + r] = _unravel(r, shape)
            expected = stop
        if expected != m:
            raise ElementGraphError(
                f"the evaluator's row map covers {expected} of its {m} rows; the rest "
                "would carry no constraint label."
            )

        # Incidence: the Jacobian sparsity pattern, grouped by row.
        jr, jc = evaluator.jacobian_structure()
        jr = np.asarray(jr, dtype=np.int64)
        jc = np.asarray(jc, dtype=np.int64)
        if jr.size and (jr.min() < 0 or jr.max() >= m or jc.min() < 0 or jc.max() >= n):
            raise ElementGraphError("Jacobian structure indexes outside the evaluator's shape.")
        row_cliques = _group_by_row(jr, jc, m)

        return cls(
            var_names=tuple(var_names),
            var_shapes=tuple(var_shapes),
            col_var=col_var,
            col_index=tuple(col_index),
            con_names=tuple(con_names),
            con_shapes=tuple(con_shapes),
            row_con=row_con,
            row_index=tuple(row_index),
            row_cliques=row_cliques,
            checks=ElementChecks(
                columns_labelled=n, rows_labelled=m, jacobian_entries=int(jr.size)
            ),
        )

    # ── sizes / labels ──────────────────────────────────────────

    @property
    def num_cols(self) -> int:
        return int(self.col_var.size)

    @property
    def num_rows(self) -> int:
        return int(self.row_con.size)

    @property
    def num_incidences(self) -> int:
        return int(sum(c.size for c in self.row_cliques))

    def col_label(self, j: int) -> str:
        """``name[i, j]`` (or ``name`` for a scalar) for column *j*."""
        return _label(self.var_names[int(self.col_var[j])], self.col_index[j])

    def row_label(self, i: int) -> str:
        """``name[i, j]`` (or ``name`` for a scalar row) for row *i*."""
        return _label(self.con_names[int(self.row_con[i])], self.row_index[i])

    def columns_of(self, var_name: str) -> np.ndarray:
        """Flat columns of the array named *var_name*."""
        try:
            p = self.var_names.index(var_name)
        except ValueError:
            raise KeyError(f"no variable named {var_name!r}") from None
        return np.flatnonzero(self.col_var == p)

    # ── structure queries ───────────────────────────────────────

    def components(
        self,
        *,
        drop_cols: Sequence[int] | np.ndarray = (),
        drop_rows: Sequence[int] | np.ndarray = (),
    ) -> tuple[np.ndarray, int]:
        """Connected components after removing *drop_cols* and *drop_rows*.

        Returns ``(block_of_col, num_blocks)``. Dropped columns get ``-1``. A
        kept column touched by no kept row is its own block, as in
        ``kernels.connected_components``; use :meth:`bearing_blocks` to count
        only the blocks that carry a row.
        """
        keep_col, cliques = self._reduced(drop_cols, drop_rows)
        local, k = kernels.connected_components(int(keep_col.size), cliques)
        out = np.full(self.num_cols, -1, dtype=np.int64)
        out[keep_col] = np.asarray(local, dtype=np.int64)
        return out, int(k)

    def bearing_blocks(
        self,
        *,
        drop_cols: Sequence[int] | np.ndarray = (),
        drop_rows: Sequence[int] | np.ndarray = (),
    ) -> int:
        """Number of components that contain at least one kept row.

        This is the "does removing these split the model?" count: a column that
        only appeared in dropped rows becomes an isolated singleton, which must
        not be mistaken for an independent subproblem.
        """
        keep_col, cliques = self._reduced(drop_cols, drop_rows)
        return int(kernels.bearing_blocks(int(keep_col.size), cliques))

    def _reduced(self, drop_cols, drop_rows) -> tuple[np.ndarray, list[list[int]]]:
        dc = np.zeros(self.num_cols, dtype=bool)
        dc[np.asarray(drop_cols, dtype=np.int64)] = True
        dr = np.zeros(self.num_rows, dtype=bool)
        dr[np.asarray(drop_rows, dtype=np.int64)] = True
        keep_col = np.flatnonzero(~dc)
        local = np.full(self.num_cols, -1, dtype=np.int64)
        local[keep_col] = np.arange(keep_col.size)
        cliques: list[list[int]] = []
        for i, clique in enumerate(self.row_cliques):
            if dr[i]:
                continue
            kept = local[clique]
            cliques.append(kept[kept >= 0].tolist())
        return keep_col, cliques

    # ── index coarsening ────────────────────────────────────────

    def coarsen(
        self,
        var_axes: Mapping[str, Sequence[int]] | None = None,
        con_axes: Mapping[str, Sequence[int]] | None = None,
    ) -> "CoarseGraph":
        """Merge the elements of each array that differ only along given axes.

        ``var_axes["x"] = (1,)`` merges every column ``x[i, t]`` with the same
        ``i`` into one supervertex ``x[i, :]``; ``con_axes`` does the same for
        constraint rows. Arrays not named keep one vertex per element; naming an
        array with every axis collapses it to one vertex (the array-level view).
        Each vertex of the result carries a weight, the number of elements it
        merged. Adjacency is inherited: a super-row touches a super-column iff
        any of their elements were incident.
        """
        var_axes = dict(var_axes or {})
        con_axes = dict(con_axes or {})
        _check_axes(var_axes, self.var_names, self.var_shapes, "variable")
        _check_axes(con_axes, self.con_names, self.con_shapes, "constraint")

        col_super, col_keys = _merge(self.col_var, self.col_index, self.var_names, var_axes)
        row_super, row_keys = _merge(self.row_con, self.row_index, self.con_names, con_axes)

        n_rows = len(row_keys)
        members: list[set[int]] = [set() for _ in range(n_rows)]
        for i, clique in enumerate(self.row_cliques):
            if clique.size:
                members[int(row_super[i])].update(col_super[clique].tolist())
        cliques = tuple(np.array(sorted(s), dtype=np.int64) for s in members)
        return CoarseGraph(
            col_keys=tuple(col_keys),
            row_keys=tuple(row_keys),
            col_of=col_super,
            row_of=row_super,
            col_weight=np.bincount(col_super, minlength=len(col_keys)).astype(np.int64),
            row_weight=np.bincount(row_super, minlength=n_rows).astype(np.int64),
            row_cliques=cliques,
        )

    def summary(self) -> str:
        _, k = self.components()
        return "\n".join(
            [
                "ElementGraph (scalar variable-constraint incidence)",
                f"  columns:    {self.num_cols} in {len(self.var_names)} arrays",
                f"  rows:       {self.num_rows} in {len(self.con_names)} constraints",
                f"  incidences: {self.num_incidences}",
                f"  blocks:     {k}",
            ]
        )


@dataclass(frozen=True)
class CoarseGraph:
    """The reduced graph produced by :meth:`ElementGraph.coarsen`.

    Attributes
    ----------
    col_keys, row_keys : tuple
        ``(array name, index tuple with merged axes set to None)`` per
        supervertex, in order of first appearance.
    col_of, row_of : np.ndarray
        Supervertex of each original column / row.
    col_weight, row_weight : np.ndarray
        Number of original elements per supervertex.
    row_cliques : tuple[np.ndarray, ...]
        Super-columns each super-row touches.
    """

    col_keys: tuple[tuple[str, tuple], ...]
    row_keys: tuple[tuple[str, tuple], ...]
    col_of: np.ndarray
    row_of: np.ndarray
    col_weight: np.ndarray
    row_weight: np.ndarray
    row_cliques: tuple[np.ndarray, ...]

    @property
    def num_cols(self) -> int:
        return len(self.col_keys)

    @property
    def num_rows(self) -> int:
        return len(self.row_keys)

    def col_label(self, s: int) -> str:
        name, idx = self.col_keys[s]
        return _label(name, idx)

    def row_label(self, s: int) -> str:
        name, idx = self.row_keys[s]
        return _label(name, idx)

    def components(self) -> tuple[np.ndarray, int]:
        """Connected components over super-columns; ``(block_of, num_blocks)``."""
        local, k = kernels.connected_components(
            self.num_cols, [c.tolist() for c in self.row_cliques]
        )
        return np.asarray(local, dtype=np.int64), int(k)

    def lift(self, super_labels: Sequence[int] | np.ndarray) -> np.ndarray:
        """Map one label per super-column to one label per original column."""
        labels = np.asarray(super_labels, dtype=np.int64)
        if labels.shape != (self.num_cols,):
            raise ValueError(
                f"expected {self.num_cols} super-column labels, got shape {labels.shape}"
            )
        out: np.ndarray = labels[self.col_of]
        return out


# ── helpers ─────────────────────────────────────────────────────


def _label(name: str, idx: tuple) -> str:
    if idx == ():
        return name
    return f"{name}[{', '.join(':' if i is None else str(i) for i in idx)}]"


def _group_by_row(jr: np.ndarray, jc: np.ndarray, m: int) -> tuple[np.ndarray, ...]:
    """Sorted unique columns per row, from COO ``(jr, jc)``."""
    empty = np.zeros(0, dtype=np.int64)
    if jr.size == 0:
        return tuple(empty for _ in range(m))
    order = np.lexsort((jc, jr))
    rs, cs = jr[order], jc[order]
    keep = np.ones(rs.size, dtype=bool)
    keep[1:] = (rs[1:] != rs[:-1]) | (cs[1:] != cs[:-1])
    rs, cs = rs[keep], cs[keep]
    bounds = np.searchsorted(rs, np.arange(m + 1))
    return tuple(cs[bounds[i] : bounds[i + 1]] for i in range(m))


def _check_axes(
    axes: Mapping[str, Sequence[int]],
    names: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    kind: str,
) -> None:
    by_name = dict(zip(names, shapes))
    for name, ax in axes.items():
        if name not in by_name:
            raise KeyError(f"no {kind} named {name!r}; known: {sorted(set(names))}")
        ndim = len(by_name[name])
        for a in ax:
            if not (0 <= int(a) < ndim):
                raise ValueError(
                    f"{kind} {name!r} has {ndim} axes (shape {by_name[name]}); "
                    f"axis {a} is out of range."
                )


def _merge(
    owner: np.ndarray,
    index: Sequence[tuple[int, ...]],
    names: Sequence[str],
    axes: Mapping[str, Sequence[int]],
) -> tuple[np.ndarray, list[tuple[str, tuple]]]:
    """Assign each element a supervertex id keyed by (owner name, kept index)."""
    merged = {name: {int(a) for a in ax} for name, ax in axes.items()}
    key_id: dict[tuple[int, tuple], int] = {}
    keys: list[tuple[str, tuple]] = []
    out = np.empty(owner.size, dtype=np.int64)
    for e in range(owner.size):
        p = int(owner[e])
        name = names[p]
        drop = merged.get(name)
        idx = index[e]
        kept = idx if not drop else tuple(None if a in drop else i for a, i in enumerate(idx))
        k = (p, kept)
        s = key_id.get(k)
        if s is None:
            s = len(keys)
            key_id[k] = s
            keys.append((name, kept))
        out[e] = s
    return out, keys


__all__ = ["CoarseGraph", "ElementChecks", "ElementGraph", "ElementGraphError"]
