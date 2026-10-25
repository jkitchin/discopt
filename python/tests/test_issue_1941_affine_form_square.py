"""#1941 item 2: squares of multi-column affine forms in the incremental McCormick path.

``uniform_relax`` lifts ``(a.x + c)**2`` to an aux column with box-dependent secant and
tangent rows and box-dependent bounds. The incremental path only patched the
single-original case (``affine_square_map``), so a model with a multi-column form
(mathopt1/2, du-opt, prob09, st_e02, ...) left those rows frozen at the root box,
failed ``_validate``, and fell back to a cold build at every node. The patch must
reproduce the cold build exactly on every box; that is what these pin.
"""

from __future__ import annotations

import numpy as np
import pytest

import discopt.modeling as dm
from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
from discopt._relax.term_classifier import classify_nonlinear_terms

pytestmark = pytest.mark.smoke


def _engine(nested: bool):
    m = dm.Model()
    x = m.continuous("x", lb=-3.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=4.0)
    if nested:
        # the form's base is itself a lifted aux (x*y), whose bounds a patched term owns
        m.minimize((x * y - 2.0 * x + 1.0) ** 2 + x - y)
    else:
        m.minimize((x + 2.0 * y - 1.0) ** 2 + (3.0 * x - y) ** 2 + x)
    m.subject_to(x + y <= 3.0)
    return IncrementalMcCormickLP(m, classify_nonlinear_terms(m))


@pytest.mark.parametrize("nested", [False, True])
def test_affine_form_square_is_patched_and_matches_cold_build(nested):
    eng = _engine(nested)
    assert eng.ok, eng.decline_reason
    assert eng.afsq_form_rows, "no affine-form square was recognised"
    rng = np.random.default_rng(1941 + nested)
    compared = mism = 0
    for _ in range(100):
        lo = np.array([rng.uniform(-3.0, 1.5), rng.uniform(-1.0, 3.5)])
        hi = np.array([rng.uniform(lo[0], 2.0), rng.uniform(lo[1], 4.0)])
        if rng.random() < 0.2:  # pinned columns: the degenerate regime
            hi[0] = lo[0]
        ap, bp, bdp = eng._patch(lo, hi)
        af, bf, bdf, _, _, _ = eng._full_build(lo, hi)
        compared += 1
        if ap.shape != af.shape or eng._rowset(ap, bp) != eng._rowset(af, bf):
            mism += 1
        elif not np.allclose(bdp, bdf, atol=1e-9, rtol=1e-9):
            mism += 1
    assert compared == 100
    assert mism == 0, f"{mism}/100 boxes mismatched"
