"""``from_mps`` / ``from_lp``: a model read from an MPS or CPLEX-LP file, in vector form.

Both formats describe ``min/max c'x + offset (+ 0.5 x'Qx)`` subject to
``row_lo <= A x <= row_hi`` and ``col_lb <= x <= col_ub``, with some columns
integer. The parsers are the in-house HiGHS ports (:func:`discopt._mps.read_mps`,
:func:`discopt._lpfile.read_lp`). This module turns their arrays into a
:class:`~discopt.modeling.core.Model` built the way the modeling API builds a
large linear model:

* **one vector variable per column type**, ``x_int`` (integer columns, binaries
  included as integers with their file bounds) and ``x_cont`` (continuous columns);
* **the rows as matrix blocks** (:meth:`Model.add_linear_constraints`), one per
  sense: ``eq`` (``row_lo == row_hi``), ``le`` (finite ``row_hi``) and ``ge``
  (finite ``row_lo``); a ranged row is in both ``le`` and ``ge``, and a row
  with no finite side is dropped (it constrains nothing);
* **the objective as array sums** over those two variables.

WHY. Going through ``.nl`` (``from_nl``) gives one scalar ``Variable`` per column
and a scalar-term objective. Every per-variable or per-term pass of a solve then
scales with the column count. Measured 2026-11-14 on MIPLIB ``tbfp-network``
(72,747 columns) with a 20 s budget: about 12 s of Python outside the engine via
``.nl``, against about 1.5 s for the same model in this form, where the engine
gets the whole budget (``docs/dev/highs-mip-replacement-plan.md`` §8.13).

COLUMN ORDER. The variables hold the columns grouped by type, so file order is
not variable order. :attr:`Model.file_columns` (a :class:`FileColumns`) keeps the
file's column names and maps a result back to file order.

REFUSED, loudly, rather than approximated:

* semi-continuous and semi-integer columns;
* an MPS quadratic objective section (QMATRIX/QUADOBJ). ``read_mps`` reports it
  but does not return the matrix, so reading on would drop it silently;
* a quadratic constraint, which neither in-house parser reads at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

__all__ = ["FileColumns", "from_lp", "from_mps"]

#: Variable names of the two column blocks.
INT_NAME = "x_int"
CONT_NAME = "x_cont"


@dataclass(frozen=True)
class FileColumns:
    """The file's columns, in file order, mapped onto the model's variables.

    ``names[k]`` is the file's ``k``-th column; it is entry ``index[k]`` of the
    variable named ``var[k]``. Stored by variable *name*, so it survives
    :meth:`Model.clone`.
    """

    names: tuple[str, ...]
    var: tuple[str, ...]
    index: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    def values(self, result) -> np.ndarray | None:
        """``result``'s point in file column order (``None`` when it has none)."""
        if result.x is None:
            return None
        out = np.empty(len(self.names), dtype=np.float64)
        for vname in set(self.var):
            mask = np.fromiter((v == vname for v in self.var), dtype=bool, count=len(self.var))
            out[mask] = np.asarray(result.x[vname], dtype=np.float64).ravel()[self.index[mask]]
        return out

    def as_dict(self, result) -> dict[str, float]:
        """``{column name: value}`` for ``result``'s point (empty when it has none)."""
        vals = self.values(result)
        if vals is None:
            return {}
        return {name: float(v) for name, v in zip(self.names, vals)}


def from_mps(path: str):
    """Read an MPS file (``.mps`` / ``.mps.gz``, free format) into a vector-form
    :class:`Model`. See the module docstring for the layout and what is refused.

    The parser reproduces HiGHS's free-format reader; a file HiGHS would hand to its
    fixed-format reader raises ``NotImplementedError``.
    """
    from discopt._mps import read_mps

    d = read_mps(path)
    if d["has_hessian"]:
        raise NotImplementedError(
            f"{path}: the MPS file has a quadratic objective section (QMATRIX/QUADOBJ), "
            "which the MPS reader does not return; reading on would drop it. Write the "
            "model as CPLEX-LP and use from_lp, or build it with add_quadratic_objective."
        )
    return _model_from_arrays(d, _model_name(path), hessian=None)


def from_lp(path: str):
    """Read a CPLEX-LP file (``.lp`` / ``.lp.gz``) into a vector-form :class:`Model`,
    quadratic objective included. See the module docstring."""
    from discopt._lpfile import read_lp

    d = read_lp(path)
    return _model_from_arrays(d, _model_name(path), hessian=d.get("hessian"))


def _model_name(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".gz", ".mps", ".lp"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base or "model"


def _model_from_arrays(d: dict, name: str, hessian) -> object:
    import discopt.modeling as dm
    from discopt.modeling.core import Constant, Model, VarBlock

    n = int(d["n"])
    integ = np.asarray(d["integrality"], dtype=np.int64)
    if np.any(integ >= 2):
        kinds = sorted({int(k) for k in integ[integ >= 2]})
        raise NotImplementedError(
            f"{name}: semi-continuous / semi-integer columns (integrality {kinds}) are "
            "not supported."
        )
    col_lb = np.asarray(d["col_lb"], dtype=np.float64)
    col_ub = np.asarray(d["col_ub"], dtype=np.float64)
    is_int = integ == 1
    ii = np.flatnonzero(is_int)
    cc = np.flatnonzero(~is_int)
    model = Model(name)
    blocks = []
    if ii.size:
        blocks.append(model.integer(INT_NAME, shape=(ii.size,), lb=col_lb[ii], ub=col_ub[ii]))
    if cc.size:
        blocks.append(model.continuous(CONT_NAME, shape=(cc.size,), lb=col_lb[cc], ub=col_ub[cc]))
    perm = np.concatenate([ii, cc]).astype(np.int64)
    # Model column k holds file column perm[k].
    xb = blocks[0] if len(blocks) == 1 else VarBlock(blocks)

    A = sp.csc_matrix(d["A"])[:, perm].tocsr()  # noqa: N806
    lo = np.asarray(d["row_lo"], dtype=np.float64)
    hi = np.asarray(d["row_hi"], dtype=np.float64)
    eq = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    le = np.isfinite(hi) & ~eq
    ge = np.isfinite(lo) & ~eq
    if n and eq.any():
        model.add_linear_constraints(A[eq], xb, "==", hi[eq], name="eq")
    if n and le.any():
        model.add_linear_constraints(A[le], xb, "<=", hi[le], name="le")
    if n and ge.any():
        model.add_linear_constraints(A[ge], xb, ">=", lo[ge], name="ge")

    c = np.asarray(d["c"], dtype=np.float64)[perm]
    pieces = [(blocks[k], sl) for k, sl in enumerate(_block_slices(ii.size, cc.size))]
    expr = Constant(np.float64(d["offset"]))
    for var, sl in pieces:
        if np.any(c[sl] != 0.0):
            expr = expr + dm.sum(c[sl] * var)
    if hessian is not None and hessian.nnz:
        expr = expr + _quadratic_terms(hessian, perm, pieces)
    (model.maximize if d["maximize"] else model.minimize)(expr)

    names = list(d["col_names"])
    if len(names) != n:
        names = [f"c{k}" for k in range(n)]
    var_of = np.empty(n, dtype=object)
    index = np.empty(n, dtype=np.int64)
    var_of[ii], index[ii] = INT_NAME, np.arange(ii.size)
    var_of[cc], index[cc] = CONT_NAME, np.arange(cc.size)
    model.file_columns = FileColumns(tuple(names), tuple(var_of.tolist()), index)
    return model


def _block_slices(n_int: int, n_cont: int) -> list[slice]:
    out = []
    if n_int:
        out.append(slice(0, n_int))
    if n_cont:
        out.append(slice(n_int, n_int + n_cont))
    return out


def _quadratic_terms(hessian, perm: np.ndarray, pieces):
    """``0.5 x'Qx`` with ``Q`` the symmetric matrix whose lower triangle ``hessian``
    holds (the LP reader's convention), in model column order, as gathered products
    over the variable blocks."""
    import discopt.modeling as dm

    low = sp.tril(sp.csc_matrix(hessian)).tocoo()
    inv = np.empty(perm.size, dtype=np.int64)
    inv[perm] = np.arange(perm.size)
    # x'Qx = sum_i Q_ii x_i^2 + 2 sum_{i>j} Q_ij x_i x_j, so 0.5 x'Qx weights a
    # diagonal entry 0.5 and an off-diagonal one 1.0.
    r = inv[low.row]
    col = inv[low.col]
    w = np.where(low.row == low.col, 0.5, 1.0) * low.data
    keep = w != 0.0
    r, col, w = r[keep], col[keep], w[keep]
    starts = [sl.start for _, sl in pieces]
    which_r = np.searchsorted(starts, r, side="right") - 1
    which_c = np.searchsorted(starts, col, side="right") - 1
    total = None
    for a, (va, sa) in enumerate(pieces):
        for b, (vb, sb) in enumerate(pieces):
            sel = (which_r == a) & (which_c == b)
            if not sel.any():
                continue
            term = dm.sum(w[sel] * va[r[sel] - sa.start] * vb[col[sel] - sb.start])
            total = term if total is None else total + term
    return total if total is not None else 0.0
