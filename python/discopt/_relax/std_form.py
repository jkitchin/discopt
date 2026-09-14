"""The one definition of the LP standard form the Rust simplex driver consumes.

Every marshaler that hands an LP to ``solve_milp_csc_py`` /
``solve_lp_warm_csc_py`` lays the columns out the same way, and this module is
where that layout is decided. It exists because the layout used to be re-derived
independently at five sites -- the four extractors in
:mod:`discopt._relax.problem_classifier` and ``_marshal_std_form`` in
:mod:`discopt.solvers.milp_simplex` -- which drifted into two *different*
standard forms reaching the same engine, so a measurement on one said nothing
about the other.

The layout
----------
Columns are ``[structural (n_struct) | one logical per row, in row order]``:
row ``r``'s logical is column ``n_struct + r``, and it is the only nonzero of
that column.

======  ==================  =========  ==========
sense   row as marshaled    logical a  logical box
======  ==================  =========  ==========
``le``  ``body + s = rhs``  ``+1``     ``[0, inf)``
``ge``  ``body - s = rhs``  ``-1``     ``[0, inf)``
``eq``  ``body + s = rhs``  ``+1``     ``[0, 0]``
======  ==================  =========  ==========

An equality's logical is *fixed at zero*: it adds a column but no freedom, so
the feasible set, the optimum and every valid bound are unchanged.

Why an equality needs a logical it cannot use
---------------------------------------------
Three parts of the engine index the basis by row and require a full ``m``-column
logical block to exist:

* ``lp/gomory.rs`` refuses to separate when ``basis.basic_vars.len() != m``
  (counter ``SepGomoryShortBasis``) -- a basis short of ``m`` columns has no
  tableau row to read a GMI cut from;
* ``lp/simplex/dual.rs``'s ``PreparedDual::prepare`` rejects the same shape
  (counter ``DualPrepRejectShape``), so every child node LP is solved cold
  instead of dual-warm-started from its parent;
* ``bnb/milp_driver.rs``'s ``BaseRows::build`` identifies row ``r``'s logical as
  column ``n_struct + r`` when substituting a cut's slack terms back to
  structural ones; without one per row the map is wrong for every row after the
  first equality and the cut is dropped (counter ``SubstDropNoSlack``).

The pre-consolidation extractors gave an equality row no column at all, so a
model with any equality row lost all three on the default ``Model.solve()``
path. Measured on ``p1183_n8`` (main, 2026-09-13): ``DualPrepAccept`` 0 of 4650
prepares, ``RootCutsGenerated`` 0, 821k phase-1 pivots.

The other pre-consolidation layout, ``_marshal_std_form``'s, split each equality
into two opposing ``<=`` rows so that both got a logical. That keeps the shape
the engine wants, at the cost of doubling the equality rows into an always-tight
opposing pair -- the degenerate configuration ``_dual_start_slack_basis``'s
docstring already blames for cold-primal stalls on lifted relaxations.

Interior-point consumers do not want this
-----------------------------------------
POUNCE's QP/IPM path takes ``A x = b`` directly and gains nothing from a logical
column; a column fixed at zero only adds a structurally singular direction to
its KKT system. The QP extractors therefore stay on the inequality-only block
(``eq_logicals=False``), which is what :func:`logical_block` returns for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: Sentinel the Rust LP layer treats as "no bound" (see CLAUDE.md: it is 1e20,
#: not ``float('inf')``). Call sites pass their own value so this module never
#: silently changes a marshaler's existing sentinel.
INF = 1e20

#: Coefficient of row ``r``'s logical column, by row sense.
LOGICAL_COEF = {"le": 1.0, "ge": -1.0, "eq": 1.0}

SENSES = ("le", "ge", "eq")


@dataclass(frozen=True)
class LogicalBlock:
    """The logical (slack) columns for one marshaled constraint block.

    Attributes
    ----------
    n_logical:
        Number of logical columns, i.e. how many columns follow the structural
        ones. ``m`` under the row-logical layout, the inequality count under the
        legacy one.
    col_of_row:
        ``(m,)`` absolute column index of each row's logical, or ``-1`` for a row
        that has none (legacy layout, equality row).
    coef, lb, ub:
        ``(n_logical,)`` coefficient and box of each logical column, ordered by
        column index.
    eq_logicals:
        Which layout this block is -- carried so a caller can record it rather
        than re-derive it.
    """

    n_logical: int
    col_of_row: np.ndarray
    coef: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    eq_logicals: bool

    def entries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """COO ``(rows, cols, vals)`` of the logical block, ready to concatenate
        onto a structural COO triple."""
        rows = np.flatnonzero(self.col_of_row >= 0).astype(np.intp)
        cols = self.col_of_row[rows].astype(np.intp)
        return rows, cols, self.coef.astype(np.float64)


def row_logicals_enabled() -> bool:
    """``DISCOPT_LP_ROW_LOGICALS`` -- give every row a logical column, including
    equalities (default OFF pending the graduation panel, CLAUDE.md §5).

    Resolved per call, through :mod:`discopt.solver_tuning`, so a test can set
    the environment variable and see it take effect.
    """
    from discopt.solver_tuning import current as _tuning_current

    return bool(_tuning_current().lp_row_logicals)


def logical_block(
    senses: Sequence[str],
    n_struct: int,
    *,
    eq_logicals: Optional[bool] = None,
    inf_value: float = INF,
) -> LogicalBlock:
    """Build the logical columns for rows with the given ``senses``.

    Parameters
    ----------
    senses:
        One of ``"le"``, ``"ge"``, ``"eq"`` per row, in the order the rows are
        marshaled.
    n_struct:
        Number of structural columns; logicals start at this index.
    eq_logicals:
        ``True`` gives every row a logical (equalities fixed at zero) and
        ``False`` only the inequalities. ``None`` (the default) asks
        :func:`row_logicals_enabled`.
    inf_value:
        Upper bound written for a free logical. Call sites pass whatever they
        wrote before consolidation (``1e20`` or ``inf``) so this function cannot
        change a marshaler's sentinel behind its back; both compare ``>= INF``
        in the Rust layer.

    Raises
    ------
    ValueError
        On an unknown sense. The senses come from constraint metadata, and a
        typo that silently produced a logical-free row is exactly the failure
        this module exists to prevent, so it refuses rather than skips.
    """
    if eq_logicals is None:
        eq_logicals = row_logicals_enabled()

    m = len(senses)
    col_of_row = np.full(m, -1, dtype=np.int64)
    coef: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    for r, sense in enumerate(senses):
        if sense not in LOGICAL_COEF:
            raise ValueError(
                f"row {r}: unknown constraint sense {sense!r}; expected one of {SENSES}"
            )
        if sense == "eq" and not eq_logicals:
            continue
        col_of_row[r] = n_struct + len(coef)
        coef.append(LOGICAL_COEF[sense])
        lb.append(0.0)
        ub.append(0.0 if sense == "eq" else float(inf_value))

    return LogicalBlock(
        n_logical=len(coef),
        col_of_row=col_of_row,
        coef=np.asarray(coef, dtype=np.float64),
        lb=np.asarray(lb, dtype=np.float64),
        ub=np.asarray(ub, dtype=np.float64),
        eq_logicals=bool(eq_logicals),
    )


def logical_is_fixed(ub_value: float) -> bool:
    """Does this logical column's upper bound mark an EQUALITY row?

    The inverse direction of :func:`logical_block`, for a consumer that holds the
    marshaled matrix and must recover which rows were equalities --
    ``solver._decompose_eq_slack_form`` and everything downstream of it (the
    MILP feasibility gate, the relaxation-dual recovery, the HiGHS marshaling).

    Sparsity alone cannot answer this under the row-logical layout: *every* row
    has a logical entry there, so a consumer that classifies "has a slack column
    => inequality" turns every equality into a one-sided inequality and stops
    checking the other direction. That is a silently weakened feasibility gate,
    so the classification must read the logical's BOX, which is what this is.
    """
    return float(ub_value) == 0.0


def logical_fixed_mask(col_ub: np.ndarray) -> np.ndarray:
    """:func:`logical_is_fixed` over a whole upper-bound vector, vectorized.

    The same predicate, kept here beside it so the scalar and array forms cannot
    drift apart. The sparse decomposition consults this once per stored nonzero,
    where a Python-level call per entry is the difference between an array index
    and a function call on every coefficient of the matrix.
    """
    mask: np.ndarray = np.asarray(col_ub, dtype=np.float64) == 0.0
    return mask
