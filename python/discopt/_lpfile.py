"""In-house CPLEX-LP-format reader (highs-parity P6): ``read_lp(path)`` returns the
model as HiGHS's ``readModel`` -> ``getLp`` (and ``getModel().hessian_``) would,
without HiGHS.

The parser is Rust (``discopt_core::lp_file``, exposed as
``discopt._rust.read_lp_text``) and ports HiGHS 1.15.1's staged LP reader stage by
stage, quirks included; the contract is an exact differential against highspy
(``python/tests/test_lp_reader.py``). Where HiGHS rejects a file (``kError``) this
raises ``ValueError``.
"""

from __future__ import annotations

import gzip

import numpy as np
import scipy.sparse as sp


def read_lp(path: str) -> dict:
    """The model in ``path`` (``.lp`` or ``.lp.gz``) as arrays.

    The keys are those of :func:`discopt._mps.read_mps` (``n``, ``m``, ``A`` (CSC,
    rows ascending within a column), ``c``, ``offset``, ``maximize``, ``col_lb``,
    ``col_ub``, ``row_lo``, ``row_hi``, ``integrality`` (0 continuous, 1 integer,
    2 semi-continuous, 3 semi-integer), ``col_names``, ``row_names`` (automatic ones
    ``HiGHS_R<i>``; empty when HiGHS clears them) and ``has_hessian``) plus
    ``objective_name`` and ``hessian``: ``None`` or HiGHS's normalised objective
    Hessian as an ``n x n`` CSC matrix holding the LOWER triangle, entries in
    HiGHS's order (diagonal first; explicit zeros on empty diagonals). The objective
    is ``c @ x + offset + 0.5 * x @ Q @ x`` with ``Q`` the symmetric matrix whose
    lower triangle this is.
    """
    from discopt._rust import read_lp_text

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        text = f.read()
    d = read_lp_text(text)
    n, m = int(d["num_col"]), int(d["num_row"])
    A = sp.csc_matrix((d["a_value"], d["a_index"], d["a_start"]), shape=(m, n))  # noqa: N806
    integ = np.asarray(d["integrality"], dtype=np.int64)
    if integ.size == 0:
        integ = np.zeros(n, dtype=np.int64)
    h = d["hessian"]
    hessian = None
    if h is not None:
        dim = int(h["dim"])
        hessian = sp.csc_matrix((h["value"], h["index"], h["start"]), shape=(dim, dim))
    return dict(
        n=n,
        m=m,
        A=A,
        c=np.asarray(d["col_cost"], dtype=np.float64),
        offset=float(d["offset"]),
        maximize=bool(d["maximize"]),
        col_lb=np.asarray(d["col_lower"], dtype=np.float64),
        col_ub=np.asarray(d["col_upper"], dtype=np.float64),
        row_lo=np.asarray(d["row_lower"], dtype=np.float64),
        row_hi=np.asarray(d["row_upper"], dtype=np.float64),
        integrality=integ,
        col_names=list(d["col_names"]),
        row_names=list(d["row_names"]),
        has_hessian=hessian is not None,
        objective_name=str(d["objective_name"]),
        hessian=hessian,
    )
