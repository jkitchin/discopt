"""In-house MPS reader (highs-parity P6): ``read_mps(path)`` returns the model as
HiGHS's ``readModel`` -> ``getLp`` would, without HiGHS.

The parser is Rust (``discopt_core::mps``, exposed as ``discopt._rust.read_mps_text``)
and reproduces HiGHS 1.15.1's free-format reader quirk for quirk; the contract is an
exact differential against highspy (``python/tests/test_mps_reader.py``). A file
HiGHS would hand to its legacy fixed-format reader raises ``NotImplementedError``
rather than being read differently.
"""

from __future__ import annotations

import gzip

import numpy as np
import scipy.sparse as sp


def read_mps(path: str) -> dict:
    """The model in ``path`` (``.mps`` or ``.mps.gz``) as arrays.

    Keys: ``n``, ``m``, ``A`` (CSC, columns in file order, entries in file order
    within a column), ``c``, ``offset``, ``maximize``, ``col_lb``, ``col_ub``,
    ``row_lo``, ``row_hi`` (infinite bounds as ``+-inf``), ``integrality`` (one
    entry per column: 0 continuous, 1 integer, 2 semi-continuous, 3 semi-integer),
    ``col_names``, ``row_names`` (empty when names repeat, as in HiGHS) and
    ``has_hessian``.
    """
    from discopt._rust import read_mps_text

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        text = f.read().decode("utf-8", errors="surrogateescape")
    d = read_mps_text(text)
    n, m = int(d["num_col"]), int(d["num_row"])
    A = sp.csc_matrix((d["a_value"], d["a_index"], d["a_start"]), shape=(m, n))
    integ = np.asarray(d["integrality"], dtype=np.int64)
    if integ.size == 0:
        integ = np.zeros(n, dtype=np.int64)
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
        has_hessian=bool(d["has_hessian"]),
    )
