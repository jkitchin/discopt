"""Retired: the matrix-form "POUNCE" MILP backend (Python-MILP-tree retirement, R11).

This module used to build a ``Model`` from the matrix data and run the Python MILP
tree (``solver._solve_milp_bb``) with POUNCE interior-point node LPs. That tree is
retired in favour of discopt's Rust MILP engine, and IPM node LPs were deliberately
not ported (no warm start across branches, no basis, no exact vertex -- the reasons
#356 took the matrix-MILP default off them;
``docs/dev/python-milp-tree-retirement.md``). ``solve_milp`` here is kept so code that
imports it, or selects ``backend="pounce"``, keeps working: it is the Rust simplex
engine, :func:`discopt.solvers.milp_simplex.solve_milp`, and says so with a
``FutureWarning``.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import MILPResult

RETIRED_MESSAGE = (
    "the matrix 'pounce' MILP backend is retired; this solve runs discopt's Rust "
    "simplex MILP engine (discopt.solvers.milp_simplex.solve_milp). Select "
    "backend='simplex' (or milp_solver='simplex') to silence this warning."
)


def solve_milp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[list[tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    max_nodes: int = 100_000,
) -> MILPResult:
    """Alias for :func:`discopt.solvers.milp_simplex.solve_milp` (same contract),
    with a ``FutureWarning`` that the POUNCE matrix MILP backend is retired."""
    from discopt.solvers.milp_simplex import solve_milp as _simplex_solve_milp

    warnings.warn(RETIRED_MESSAGE, FutureWarning, stacklevel=2)
    return _simplex_solve_milp(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        integrality=integrality,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        max_nodes=max_nodes,
    )
