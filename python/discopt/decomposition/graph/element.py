"""Element-level incidence graph and index coarsening.

:class:`~discopt.decomposition.graph.base.ModelGraph` is built from
variable *names* (``variable_names_in``), which map every ``x[s, t]`` back to the whole array
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

**Constraints the evaluator does not emit.** Disjunctions, SOS sets, indicator
and logical constraints are not rows of the NLP evaluator, but they still tie
variables together. Each one becomes a single *pseudo-row* touching every
column of every array it references. That over-approximates (an SOS over
``x[0], x[1]`` merges all of ``x``), which errs toward fewer blocks, the safe
direction; ignoring them would report coupled variables as independent. A
constraint type this module cannot walk is refused
(:class:`UnsupportedConstraintError`) rather than skipped.

**Scope.** Like the array-level layer, the graph is built from constraints
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


class UnsupportedConstraintError(ElementGraphError):
    """A model constraint of a kind whose variables this module cannot enumerate."""


def _constraint_variables(con) -> list:
    """Every ``Variable`` a constraint of any kind references (for pseudo-rows).

    Mirrors ``Model._iter_owned_leaves``'s per-constraint case analysis, which
    is the walk the solver's ownership guard relies on to see every leaf.
    """
    from discopt.modeling.core import (
        Constraint,
        Variable,
        _DisjunctiveConstraint,
        _IndicatorConstraint,
        _iter_model_leaves,
        _logical_backing_vars,
        _LogicalConstraint,
        _SOSConstraint,
    )

    def leaves(c):
        if isinstance(c, Constraint):
            yield from _iter_model_leaves(c.body)
        elif isinstance(c, _IndicatorConstraint):
            yield from _iter_model_leaves(c.indicator)
            yield from leaves(c.constraint)
        elif isinstance(c, _DisjunctiveConstraint):
            for disjunct in c.disjuncts:
                for sub in disjunct:
                    yield from leaves(sub)
        elif isinstance(c, _SOSConstraint):
            yield from _iter_model_leaves(*c.variables)
        elif isinstance(c, _LogicalConstraint):
            yield from _iter_model_leaves(*_logical_backing_vars(c.expression))
        elif hasattr(c, "body"):
            yield from _iter_model_leaves(c.body)
        else:
            raise UnsupportedConstraintError(
                f"cannot enumerate the variables of a {type(c).__name__}; treating it "
                "as absent would report the variables it couples as independent."
            )

    out: dict[str, Any] = {}
    for leaf in leaves(con):
        if isinstance(leaf, Variable):
            out.setdefault(leaf.name, leaf)
    return list(out.values())


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
    pseudo_rows: int = 0


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
        One entry per source constraint: first the evaluator's row map, in row
        order, then one pseudo-row entry per model constraint the evaluator does
        not emit (see the module docstring). ``con_names[k]`` is the
        constraint's name, or ``"c{k}"`` when unnamed.
    con_model_index : tuple[int | None, ...]
        ``con_model_index[k]`` is the position of entry ``k`` in
        ``model._constraints``, or ``None`` for rows with no model constraint
        object (fast-builder rows, materialized fresh by the evaluator).
    con_is_algebraic : tuple[bool, ...]
        False for pseudo-row entries.
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
    con_model_index: tuple[int | None, ...]
    con_is_algebraic: tuple[bool, ...]
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
        than guessed around. Raises :class:`UnsupportedConstraintError` for a
        constraint kind whose variables cannot be enumerated.
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
        model_index_of = {id(c): i for i, c in enumerate(model._constraints)}
        con_names: list[str] = []
        con_shapes: list[tuple[int, ...]] = []
        con_model_index: list[int | None] = []
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
            con_model_index.append(model_index_of.get(id(con)))
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
        row_cliques = list(_group_by_row(jr, jc, m))
        con_is_algebraic = [True] * len(con_names)

        # Pseudo-rows: model constraints the evaluator does not emit.
        covered = {i for i in con_model_index if i is not None}
        var_cols: dict[str, np.ndarray] = {}
        offset = 0
        for v in model._variables:
            size = int(v.size) if v.shape != () else 1
            var_cols[v.name] = np.arange(offset, offset + size, dtype=np.int64)
            offset += size
        row_con_list = row_con.tolist()
        pseudo = 0
        for i, c in enumerate(model._constraints):
            if i in covered:
                continue
            cols = [var_cols[v.name] for v in _constraint_variables(c) if v.name in var_cols]
            clique = np.unique(np.concatenate(cols)) if cols else np.zeros(0, np.int64)
            k = len(con_names)
            name = getattr(c, "name", None)
            con_names.append(str(name) if name else f"{type(c).__name__.lstrip('_')}{i}")
            con_shapes.append(())
            con_model_index.append(i)
            con_is_algebraic.append(False)
            row_con_list.append(k)
            row_index.append(())
            row_cliques.append(clique)
            pseudo += 1
        row_con = np.asarray(row_con_list, dtype=np.int64)

        return cls(
            var_names=tuple(var_names),
            var_shapes=tuple(var_shapes),
            col_var=col_var,
            col_index=tuple(col_index),
            con_names=tuple(con_names),
            con_shapes=tuple(con_shapes),
            con_model_index=tuple(con_model_index),
            con_is_algebraic=tuple(con_is_algebraic),
            row_con=row_con,
            row_index=tuple(row_index),
            row_cliques=tuple(row_cliques),
            checks=ElementChecks(
                columns_labelled=n,
                rows_labelled=m,
                jacobian_entries=int(jr.size),
                pseudo_rows=pseudo,
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

    def constraint_cliques(self) -> list[list[int]]:
        """Columns per constraint entry (the union over its rows)."""
        members: list[set[int]] = [set() for _ in self.con_names]
        for i, clique in enumerate(self.row_cliques):
            members[int(self.row_con[i])].update(clique.tolist())
        return [sorted(s) for s in members]

    def bridge_constraints(self) -> list[int]:
        """Model constraints whose sole removal splits the element graph.

        The element-level analogue of the array-level bridge scan: a constraint
        is reported when dropping all of its rows raises the number of
        constraint-bearing components. Linear time (:func:`separating_cliques`),
        so unlike the array-level scan it needs no budget. Only algebraic
        constraints that are ``model._constraints`` entries are reported: a
        pseudo-row or a fast-builder row is never offered for dualization.
        Like the array-level scan, a graph that is already split reports none.
        """
        found = separating_cliques(self.num_cols, self.constraint_cliques())
        return sorted(
            int(mi)
            for k in found
            if self.con_is_algebraic[k] and (mi := self.con_model_index[k]) is not None
        )

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


def separating_cliques(n: int, cliques: Sequence[Sequence[int]]) -> set[int]:
    """Cliques whose removal raises the number of clique-bearing components.

    Same answer as ``kernels.bridge_cliques_status`` (which re-runs a component
    count per clique, ``O(m·(n+E))``), in one ``O(n + m + E)`` pass: an
    iterative Tarjan DFS over the bipartite column/clique graph, where clique
    ``k`` is a cut vertex that separates bearing pieces. For each clique node
    the pieces left by its removal are the DFS children ``w`` with
    ``low[w] >= disc[k]`` plus, unless it is a DFS root, the rest of its
    component; a piece *bears* when it holds at least one other clique. Returns
    the empty set when the graph already has two or more bearing components,
    matching the kernel ("no single coupling clique is needed").
    """
    m = len(cliques)
    total = n + m
    adj: list[list[int]] = [[] for _ in range(total)]
    for k, clique in enumerate(cliques):
        node = n + k
        for j in clique:
            adj[node].append(int(j))
            adj[int(j)].append(node)

    disc = [-1] * total
    low = [0] * total
    tparent = [-1] * total
    sub_cons = [0] * total  # clique nodes in the DFS subtree (inclusive)
    found: set[int] = set()
    bearing_components = 0
    timer = 0
    for root in range(n, total):  # start only from cliques: bearing components
        if disc[root] != -1 or not adj[root]:
            continue  # visited, or a clique over no column (bears nothing)
        bearing_components += 1
        comp_nodes = [root]
        disc[root] = low[root] = timer
        timer += 1
        stack: list[tuple[int, int]] = [(root, 0)]
        while stack:
            v, it = stack[-1]
            if it < len(adj[v]):
                stack[-1] = (v, it + 1)
                w = adj[v][it]
                if disc[w] == -1:
                    tparent[w] = v
                    disc[w] = low[w] = timer
                    timer += 1
                    comp_nodes.append(w)
                    stack.append((w, 0))
                elif w != tparent[v]:
                    low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if v >= n:
                    sub_cons[v] += 1
                p = tparent[v]
                if p != -1:
                    low[p] = min(low[p], low[v])
                    sub_cons[p] += sub_cons[v]
        comp_cons = sub_cons[root]
        children: dict[int, list[int]] = {}
        for w in comp_nodes:
            if tparent[w] != -1:
                children.setdefault(tparent[w], []).append(w)
        for v in comp_nodes:
            if v < n or len(cliques[v - n]) < 2:
                continue
            pieces = 0
            separated = 0
            for w in children.get(v, ()):
                if low[w] >= disc[v]:
                    separated += sub_cons[w]
                    if sub_cons[w] > 0:
                        pieces += 1
            # Whatever is not in a separated subtree stays connected through the
            # parent side (empty for a DFS root: every root child is separated).
            if comp_cons - 1 - separated > 0:
                pieces += 1
            if pieces >= 2:
                found.add(v - n)
    if bearing_components >= 2:
        return set()
    return found


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


__all__ = [
    "CoarseGraph",
    "ElementChecks",
    "ElementGraph",
    "ElementGraphError",
    "UnsupportedConstraintError",
]
