"""EP3 (#632): the patch-table node path for the uniform factorable engine.

``UniformPatchTable`` restores the incremental fast path
(``MccormickLPRelaxer._try_incremental_node``) for engine-shaped models. Its
``_patch`` regenerates the node relaxation through the (EP1-cached) engine build,
so the per-node solve skips the separation chain (inherited root pool substitutes)
and warm-starts — byte-identical to the cold ``build_milp_relaxation`` by
construction, verified here row-for-row at several reachable child boxes, and only
engaged when the lifted column layout is box-stable (else cold fallback).

The cheap closed-form coefficient refresh ``IncrementalMcCormickLP`` does for
bare-variable bilinear/monomial rows is NOT byte-reproducible for the engine's
folded / affine-argument atoms (their coefficients are floating-point results of
``evaluate_interval`` over the reconstructed DAG), so that table stays ``ok=False``
on engine-shaped models; this one regenerates through the build itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import numpy as np
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt._jax.incremental_mccormick import UniformPatchTable
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import from_nl

_NL = Path(__file__).parent / "data" / "minlplib_nl"

# Engine-shaped, finite-root instances the patch table validates on (a mix of
# product / power / univariate_call / composite families).
_ENGAGING = [
    "nvs09",
    "ex1226",
    "st_miqp2",
    "st_miqp3",
    "st_miqp4",
    "dispatch",
    "nvs02",
    "nvs13",
    "nvs14",
    "nvs21",
    "ex1225",
    "st_e38",
]


def _model(name):
    return from_nl(str(_NL / f"{name}.nl"))


def _child_boxes(lb, ub, n=4):
    width = ub - lb
    cols = [c for c in range(lb.size) if np.isfinite(width[c]) and width[c] > 1e-9]
    boxes = []
    for i in range(n):
        if not cols:
            break
        col = cols[i % len(cols)]
        clb, cub = lb.copy(), ub.copy()
        mid = 0.5 * (clb[col] + cub[col])
        if i % 2 == 0:
            cub[col] = mid
        else:
            clb[col] = mid
        boxes.append((clb, cub))
    return boxes


def test_uniform_patch_engages_on_engine_models():
    """The default relaxer wires ``UniformPatchTable`` as the fast path for
    engine-shaped models (the closed-form table never validates on these)."""
    engaged = 0
    for name in _ENGAGING:
        pt = UniformPatchTable(_model(name))
        if pt.ok:
            engaged += 1
    # Require a solid majority so a family regression is caught (not all listed
    # instances are guaranteed finite-root/box-stable on every platform).
    assert engaged >= 8, f"only {engaged}/{len(_ENGAGING)} engine models engaged"


def test_relaxer_selects_patch_table_for_nvs09():
    r = MccormickLPRelaxer(_model("nvs09"))
    assert isinstance(r._inc, UniformPatchTable)
    assert r._inc.ok


@pytest.mark.parametrize("name", _ENGAGING)
def test_patched_node_lp_byte_equals_cold_build(name):
    """The EP3 ``_validate`` invariant, surfaced as a test: for each engaging
    instance the patched node LP (matrix + rhs + column bounds) is byte-identical
    to the cold ``build_milp_relaxation`` at several reachable child boxes."""
    model = _model(name)
    pt = UniformPatchTable(model)
    if not pt.ok:
        pytest.skip(f"{name} does not engage (unbounded root / unstable layout)")
    terms = classify_nonlinear_terms(model)
    lb, ub = flat_variable_bounds(model)
    checked = 0
    for clb, cub in _child_boxes(lb, ub):
        Ap, bp, bdp = pt._patch(clb, cub)
        cold, _ = build_milp_relaxation(
            model, terms, DiscretizationState(), bound_override=(clb, cub)
        )
        import scipy.sparse as sp

        Af = np.asarray(sp.csr_matrix(cold._A_ub).todense(), dtype=np.float64)
        bf = np.asarray(cold._b_ub, dtype=np.float64).ravel()
        bdf = np.asarray(cold._bounds, dtype=np.float64)
        assert Ap.shape == Af.shape
        assert np.array_equal(Ap, Af), f"{name}: matrix differs from cold build"
        assert np.array_equal(bp, bf), f"{name}: rhs differs from cold build"
        assert np.array_equal(bdp, bdf), f"{name}: column bounds differ from cold build"
        checked += 1
    assert checked >= 1


def test_unbounded_root_declines_engagement():
    """A model whose ROOT relaxation has no valid objective bound (unbounded box)
    must NOT engage — the patch table probes at the root, so an invalid root build
    yields ``ok=False`` and the trusted cold path runs unchanged."""
    for name in ("alan", "fac2"):
        lb, ub = flat_variable_bounds(_model(name))
        if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
            pt = UniformPatchTable(_model(name))
            assert not pt.ok, f"{name} has an unbounded root but engaged"


def test_patch_table_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DISCOPT_INCREMENTAL_MC", "0")
    assert MccormickLPRelaxer(_model("nvs09"))._inc is None


def test_patched_node_bound_is_sound_lower_bound():
    """Where it engages, the fast-path node LP bound must be a valid lower bound
    (<= the cold-path bound at the same box, since the cold path additionally
    separates per-node cuts the fast path defers to the inherited pool)."""
    model = _model("nvs09")
    lb, ub = flat_variable_bounds(model)
    fast = MccormickLPRelaxer(model)
    assert isinstance(fast._inc, UniformPatchTable)
    r_fast = fast.solve_at_node(lb.copy(), ub.copy())
    cold = MccormickLPRelaxer(model)
    cold._inc = None
    r_cold = cold.solve_at_node(lb.copy(), ub.copy())
    assert r_fast.status == "optimal" and r_cold.status == "optimal"
    assert r_fast.lower_bound is not None and np.isfinite(r_fast.lower_bound)
    # Valid lower bound, never over-tightening the cold McCormick bound.
    assert r_fast.lower_bound <= r_cold.lower_bound + 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
