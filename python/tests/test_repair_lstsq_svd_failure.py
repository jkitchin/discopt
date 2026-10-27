"""``repair_point`` must not crash when numpy's least-squares SVD fails to converge.

On MINLPLib ``pooling_bental5tp`` a default ``Model.solve()`` raised
``LinAlgError: SVD did not converge in Linear Least Squares`` from
``feasibility._newton_repair``: numpy's ``gelsd`` failed on a finite, well-scaled,
rank-deficient 68x72 repair Jacobian (rank 55, entries up to 234) that SciPy's
LAPACK drivers solve to a residual ~1e-23. The system is the captured one.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
from discopt.validation.feasibility import _min_norm_lstsq

DATA = os.path.join(os.path.dirname(__file__), "data")


@pytest.mark.smoke
def test_the_captured_repair_system_is_solved():
    d = np.load(os.path.join(DATA, "pooling_bental5tp_repair_lstsq.npz"))
    A, b = d["A"], d["b"]
    assert A.shape == (68, 72) and np.isfinite(A).all()
    x = _min_norm_lstsq(A, b)
    assert np.isfinite(x).all()
    assert np.linalg.norm(A @ x - b) <= 1e-9 * (1.0 + np.linalg.norm(b))


@pytest.mark.smoke
def test_a_solvable_system_is_numpys_answer_exactly():
    rng = np.random.default_rng(0)
    A, b = rng.normal(size=(5, 7)), rng.normal(size=5)
    assert np.array_equal(_min_norm_lstsq(A, b), np.linalg.lstsq(A, b, rcond=None)[0])


@pytest.mark.slow
@pytest.mark.correctness
def test_pooling_bental5tp_solves_to_its_optimum():
    from discopt.modeling.core import from_nl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = from_nl(os.path.join(DATA, "minlplib", "pooling_bental5tp.nl")).solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-3500.0, abs=1e-4)
