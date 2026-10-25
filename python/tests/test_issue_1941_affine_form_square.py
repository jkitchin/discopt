"""#1941 item 2: squares of multi-column affine forms in the incremental McCormick path.

``uniform_relax`` lifts ``(a.x + c)**2`` to an aux column with box-dependent secant and
tangent rows and box-dependent bounds. The incremental path only patched the
single-original case (``affine_square_map``), so a model with a multi-column form
(mathopt1/2, du-opt, prob09, st_e02, ...) left those rows frozen at the root box,
failed ``_validate``, and fell back to a cold build at every node. The patch must
reproduce the cold build exactly on every box; that is what these pin.

Admission alone made the fast path weaker than the cold node build it replaces: the
cold build also adds the separable objective floor. ``DISCOPT_INCREMENTAL_WIDEN``
therefore carries both parts, and is off by default until its panel passes.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
from discopt._relax.term_classifier import classify_nonlinear_terms

pytestmark = pytest.mark.smoke


@pytest.fixture
def widen(monkeypatch):
    monkeypatch.setenv("DISCOPT_INCREMENTAL_WIDEN", "1")


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
def test_affine_form_square_is_patched_and_matches_cold_build(widen, nested):
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


def test_flag_off_declines_as_before(monkeypatch):
    monkeypatch.setenv("DISCOPT_INCREMENTAL_WIDEN", "0")
    eng = _engine(nested=False)
    assert not eng.ok and not eng.afsq_form_rows


def _double_well():
    m = dm.Model("well")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize((x**2 - 1) ** 2 + y**2 + 0.3 * x)
    m.subject_to(x + y >= -2)
    return m


def test_fast_path_root_bound_matches_the_cold_node_build(widen):
    """The admitted fast path must carry the cold build's objective floor: without
    it the root bound was -0.5875 against the cold path's -0.3054."""
    from discopt._relax.mccormick_lp import MccormickLPRelaxer
    from discopt._relax.model_utils import flat_variable_bounds

    m = _double_well()
    lb, ub = flat_variable_bounds(m)
    relaxer = MccormickLPRelaxer(m)
    assert relaxer._inc is not None, "the double well should take the fast path"
    assert relaxer._inc._objective_floor_row(lb, ub) is not None
    fast = relaxer._try_incremental_node(lb, ub, None)
    inc, relaxer._inc = relaxer._inc, None
    cold = relaxer.solve_at_node(lb, ub, separate=False)
    relaxer._inc = inc
    assert fast.status == cold.status == "optimal"
    assert fast.lower_bound == pytest.approx(cold.lower_bound, abs=1e-7)
    assert fast.lower_bound == pytest.approx(-0.3054284845, abs=1e-6)


@pytest.mark.parametrize("on", ["0", "1"])
def test_double_well_tree_is_unchanged_by_the_flag(monkeypatch, on):
    monkeypatch.setenv("DISCOPT_INCREMENTAL_WIDEN", on)
    r = _double_well().solve(time_limit=20)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-0.305428483743916, abs=1e-6)
    assert r.node_count == 3
