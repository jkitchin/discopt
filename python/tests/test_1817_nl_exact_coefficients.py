"""#1817: the ``.nl`` reader snapped coefficients within 1e-15 of +-1 to +-1.

Same class as #1813 (fixed in the fast-API builder), on the main corpus path:
``nl_parser.rs`` rebuilt a linear term ``c * x`` as ``x`` (or ``-x``) whenever
``|c -+ 1| < 1e-15``, and treated a nonlinear part with ``|v| < 1e-15`` as zero.
The presolve rebuilders (``coefficient_strengthening``, ``polynomial``,
``redundancy``) did the same and are covered by Rust unit tests next to them.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.problem_classifier import extract_qp_data
from discopt.export.nl import to_nl
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.smoke

A = 1.0 - 2.0**-52  # within 1e-15 of 1: was snapped to 1
B = 1e-17


def _dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def test_nl_round_trip_keeps_coefficients_exactly(tmp_path):
    m = dm.Model("nl")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(A * x + B * y == 0.5, name="r0")
    m.subject_to(-A * x - y == -0.5, name="r1")
    m.minimize(A * x - A * y + B * x)
    path = tmp_path / "m.nl"
    to_nl(m, path)
    text = path.read_text()
    assert repr(A) in text  # the writer itself is exact; the reader is under test

    back = from_nl(str(path))
    qd = extract_qp_data(back)
    Aeq = _dense(qd.A_eq)
    rows = {tuple(r) for r in Aeq[:, :2].tolist()}
    assert (A, B) in rows
    assert (-A, -1.0) in rows
    np.testing.assert_array_equal(np.asarray(qd.c)[:2], [A + B, -A])
