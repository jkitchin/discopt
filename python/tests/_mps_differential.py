"""Exact differential between discopt's MPS / LP readers and HiGHS's (``readModel``
-> ``getLp`` and ``getModel().hessian_``), shared by ``test_mps_reader.py``,
``test_lp_reader.py`` and the corpus sweeps
(``docs/dev/data/highs-parity/{mps,lp}_reader/``).

Both sides are normalized to the same canonical arrays: the matrix as CSC with
duplicates summed and explicit zeros kept exactly as each reader reports them, and
every infinite bound as ``+-inf``. The objective Hessian (LP reader) is compared raw:
the same column starts, row indices and values IN THE SAME ORDER (HiGHS's
``normaliseHessian`` order is part of the contract). Nothing is compared with a
tolerance: a reader that agrees with HiGHS agrees bit for bit.
"""

from __future__ import annotations

import gzip
import os
import shutil
import tempfile

import numpy as np
import scipy.sparse as sp


def highs_arrays(path: str) -> dict:
    """HiGHS's reading of ``path`` as canonical arrays."""
    import highspy

    tmpdir = None
    src = path
    if path.endswith(".gz"):
        tmpdir = tempfile.mkdtemp(prefix="mpsdiff_")
        # HiGHS picks its reader by extension: keep the one under the ".gz".
        src = os.path.join(tmpdir, "model" + os.path.splitext(path[: -len(".gz")])[1])
        with gzip.open(path, "rb") as f, open(src, "wb") as g:
            shutil.copyfileobj(f, g)
    try:
        h = highspy.Highs()
        h.silent()
        if h.readModel(src) == highspy.HighsStatus.kError:
            raise ValueError(f"HiGHS could not read {path!r}")
        lp = h.getLp()
        hs = h.getModel().hessian_
        hessian = None
        if hs.dim_:
            hessian = sp.csc_matrix(
                (
                    np.array(hs.value_, dtype=np.float64),
                    np.array(hs.index_, dtype=np.int64),
                    np.array(hs.start_, dtype=np.int64),
                ),
                shape=(hs.dim_, hs.dim_),
            )
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    vals = np.array(a.value_, dtype=np.float64)
    idx = np.array(a.index_, dtype=np.int64)
    start = np.array(a.start_, dtype=np.int64)
    if a.format_ == highspy.MatrixFormat.kRowwise:
        A = sp.csr_matrix((vals, idx, start), (m, n)).tocsc()
    else:
        A = sp.csc_matrix((vals, idx, start), (m, n))
    integ = np.array([int(t) for t in lp.integrality_], dtype=np.int64)
    if integ.size == 0:
        integ = np.zeros(n, dtype=np.int64)
    return dict(
        n=n,
        m=m,
        A=A,
        c=np.array(lp.col_cost_, dtype=np.float64),
        offset=float(lp.offset_),
        maximize=lp.sense_ == highspy.ObjSense.kMaximize,
        col_lb=np.array(lp.col_lower_, dtype=np.float64),
        col_ub=np.array(lp.col_upper_, dtype=np.float64),
        row_lo=np.array(lp.row_lower_, dtype=np.float64),
        row_hi=np.array(lp.row_upper_, dtype=np.float64),
        integrality=integ,
        col_names=list(lp.col_names_),
        row_names=list(lp.row_names_),
        hessian=hessian,
    )


def _canon_matrix(A: sp.spmatrix):
    A = sp.csc_matrix(A, dtype=np.float64)
    A.sort_indices()
    return A.indptr, A.indices, A.data


def differences(ours: dict, ref: dict) -> list[str]:
    """Every field on which ``ours`` differs from ``ref`` (empty: identical)."""
    out = []
    for k in ("n", "m", "offset", "maximize"):
        # A NaN offset survives a read on both sides (LP files): equal there.
        both_nan = k == "offset" and np.isnan(ours[k]) and np.isnan(ref[k])
        if ours[k] != ref[k] and not both_nan:
            out.append(f"{k}: {ours[k]!r} vs {ref[k]!r}")
    if out:
        return out
    for k in ("c", "col_lb", "col_ub", "row_lo", "row_hi", "integrality"):
        a, b = np.asarray(ours[k]), np.asarray(ref[k])
        # NaN survives a read on both sides (HiGHS keeps a NaN cost): equal there.
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            bad = np.flatnonzero(a != b) if a.shape == b.shape else []
            out.append(f"{k}: differs at {list(bad[:5])} (shape {a.shape} vs {b.shape})")
    for k in ("col_names", "row_names"):
        if list(ours[k]) != list(ref[k]):
            out.append(f"{k} differ")
    pa, ia, va = _canon_matrix(ours["A"])
    pb, ib, vb = _canon_matrix(ref["A"])
    same = np.array_equal(pa, pb) and np.array_equal(ia, ib)
    if not (same and np.array_equal(va, vb, equal_nan=True)):
        out.append(f"A differs (nnz {va.size} vs {vb.size})")
    if "hessian" in ours:
        out += _hessian_differences(ours["hessian"], ref["hessian"])
    return out


def _hessian_differences(ha, hb) -> list[str]:
    """Raw comparison of two HiGHS-normalised Hessians (order included)."""
    if ha is None or hb is None:
        return [] if ha is None and hb is None else [f"hessian: {ha is None} vs {hb is None}"]
    if ha.shape != hb.shape:
        return [f"hessian shape {ha.shape} vs {hb.shape}"]
    same = np.array_equal(ha.indptr, hb.indptr) and np.array_equal(ha.indices, hb.indices)
    if not (same and np.array_equal(ha.data, hb.data, equal_nan=True)):
        return [f"hessian differs (nnz {ha.data.size} vs {hb.data.size})"]
    return []
