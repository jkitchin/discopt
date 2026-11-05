"""A persistent in-house MILP, edited in place and re-solved (highs-parity P3.1).

The MILP IIS engine (``discopt._iis_milp._Engine``) builds one model and toggles
elements -- a row's bounds, a column's bounds, a column's integrality -- solving after
each toggle. This is the in-house counterpart of the one ``Highs`` object it holds
today, over the Rust MILP driver.

Layout. The caller's rows are ranged, ``row_lo <= A x <= row_hi``; each row ``i``
gets one logical column ``s_i`` with ``A_i x - s_i = 0`` and ``s_i`` boxed
``[row_lo_i, row_hi_i]``. So :meth:`MilpHandle.change_row_bounds` is a bound change
on that logical, and dropping a row is ``(-inf, inf)``. Infinite sides are passed as
the Rust LP layer's ``1e20`` sentinel (CLAUDE.md "Key Constraints").

Every solve is a fresh run of the same driver ``solve_milp_csc_py`` runs, on the
handle's current arrays, under the same options for the same keywords
(``discopt._rust.MilpHandle``): nothing is carried between solves but the data.
``python/tests/test_highs_parity_p31_milp_handle.py`` checks that bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

#: The Rust LP layer's infinity sentinel.
RUST_INF = 1e20


def _rust_side(v: np.ndarray) -> np.ndarray:
    """``±inf`` (and anything past the sentinel) as ``±1e20``; NaN is refused."""
    v = np.asarray(v, dtype=np.float64)
    if np.isnan(v).any():
        raise ValueError("MilpHandle: NaN bound")
    return np.clip(v, -RUST_INF, RUST_INF)


@dataclass
class HandleResult:
    """One solve: ``x`` over the structural columns (the caller's variables)."""

    status: str
    x: np.ndarray
    objective: float
    bound: float
    nodes: int
    lp_iters: int


class MilpHandle:
    """``min cᵀx + obj_const`` s.t. ``row_lo <= A x <= row_hi``, ``col_lb <= x <=
    col_ub``, ``x_j`` integer where ``is_int[j]`` -- held in Rust, edited in place."""

    def __init__(
        self,
        c: Any,
        A: Any,
        row_lo: Any,
        row_hi: Any,
        col_lb: Any,
        col_ub: Any,
        is_int: Any,
        obj_const: float = 0.0,
    ) -> None:
        from discopt._rust import MilpHandle as _RustHandle

        A = sp.csc_matrix(A, dtype=np.float64)
        m, n_struct = A.shape
        c = np.asarray(c, dtype=np.float64).ravel()
        is_int = np.asarray(is_int, dtype=bool).ravel()
        for name, arr, want in (
            ("c", c, n_struct),
            ("col_lb", np.ravel(col_lb), n_struct),
            ("col_ub", np.ravel(col_ub), n_struct),
            ("is_int", is_int, n_struct),
            ("row_lo", np.ravel(row_lo), m),
            ("row_hi", np.ravel(row_hi), m),
        ):
            if arr.shape != (want,):
                raise ValueError(f"MilpHandle: {name} has shape {arr.shape}, expected ({want},)")
        full = sp.hstack([A, -sp.identity(m, format="csc")], format="csc")
        full.sort_indices()
        self.m, self.n_struct, self.n = m, n_struct, n_struct + m
        self._h = _RustHandle(
            np.concatenate([c, np.zeros(m)]),
            m,
            self.n,
            full.indptr.astype(np.int64),
            full.indices.astype(np.int64),
            full.data.astype(np.float64),
            np.zeros(m),
            _rust_side(np.concatenate([np.ravel(col_lb), np.ravel(row_lo)])),
            _rust_side(np.concatenate([np.ravel(col_ub), np.ravel(row_hi)])),
            np.flatnonzero(is_int).astype(np.int64),
            n_struct,
            float(obj_const),
        )

    # -- edits ------------------------------------------------------------------
    def change_col_bounds(self, j: int, lo: float, hi: float) -> None:
        """Set structural column ``j``'s box."""
        if not 0 <= j < self.n_struct:
            raise IndexError(f"column {j} out of range (n_struct={self.n_struct})")
        lo_r, hi_r = _rust_side(np.array([lo, hi]))
        self._h.change_col_bounds(int(j), float(lo_r), float(hi_r))

    def change_row_bounds(self, i: int, lo: float, hi: float) -> None:
        """Set row ``i``'s range; ``(-inf, inf)`` drops the row."""
        if not 0 <= i < self.m:
            raise IndexError(f"row {i} out of range (m={self.m})")
        lo_r, hi_r = _rust_side(np.array([lo, hi]))
        self._h.change_col_bounds(self.n_struct + int(i), float(lo_r), float(hi_r))

    def configure(self, row_lo: Any, row_hi: Any, col_lb: Any, col_ub: Any, is_int: Any) -> None:
        """Replace every row range, column box and integrality flag at once (the IIS
        engine's per-probe configuration): two Rust calls, no per-element Python."""
        lb = np.concatenate([np.ravel(col_lb), np.ravel(row_lo)])
        ub = np.concatenate([np.ravel(col_ub), np.ravel(row_hi)])
        if lb.shape != (self.n,):
            raise ValueError(f"configure: {lb.size} bounds for {self.n} columns")
        self._h.set_col_bounds_all(_rust_side(lb), _rust_side(ub))
        self._h.set_integrality_all(np.asarray(is_int, dtype=bool).ravel())

    def change_integrality(self, j: int, is_int: bool) -> None:
        """Make structural column ``j`` integer or continuous."""
        self._h.change_integrality(int(j), bool(is_int))

    def set_solution(self, x: Optional[Any]) -> None:
        """Seed (``x`` over the structural columns) or clear (``None``) the incumbent
        the next solves start from; the driver validates it, never trusts it."""
        self._h.set_solution(None if x is None else np.asarray(x, dtype=np.float64).ravel())

    # -- solves -----------------------------------------------------------------
    def solve(
        self,
        *,
        time_limit: Optional[float] = None,
        presolve: bool = True,
        max_nodes: int = 1_000_000,
        **options: Any,
    ) -> HandleResult:
        """Solve the current problem. ``options`` are ``solve_milp_csc_py``'s keywords
        (``gap_tol``, ``heuristics``, ...), with its defaults."""
        status, x, obj, bound, nodes, iters = self._h.solve(
            max_nodes=int(max_nodes), presolve=bool(presolve), time_limit_s=time_limit, **options
        )
        return HandleResult(status, np.asarray(x), float(obj), float(bound), int(nodes), int(iters))

    def presolve_only(self) -> bool:
        """``True`` when the driver's root preparation and presolve prove the current
        box empty (a ``presolve=True`` solve's zero-node ``infeasible``); ``False``
        decides nothing."""
        return bool(self._h.presolve_only())

    def std_form(self) -> dict:
        """The arrays a fresh ``solve_milp_csc_py`` call needs to reproduce the
        current problem (for comparison; the handle does not read this back)."""
        lb, ub = self._h.col_bounds()
        return {"lb": np.asarray(lb), "ub": np.asarray(ub),
                "integer_cols": np.asarray(self._h.integer_cols(), dtype=np.int64)}  # fmt: skip
