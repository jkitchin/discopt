"""Regression lock for issue #154, increment 1: root FBBT must propagate *finite*
bounds through the division/sqrt auxiliary-variable chains of bucket-2 instances.

Several bucket-2 instances (``nvs05``, ``nvs22``, ``chance``) introduce auxiliary
variables for nonlinear sub-expressions — e.g. ``x4 = c/(x0*x1)`` and
``x6 = sqrt(...)`` — which the ``.nl`` model declares with the default
``[-inf, +inf]`` bounds. Issue #154 originally proposed a Python-side FBBT pass to
finitize those aux bounds so the reciprocal/sqrt envelope lift (#157) has finite
factor bounds to work with (the lift abstains on non-finite bounds).

That finitization is already delivered by the Rust FBBT layer (PR #135): after
``tighten_root_bounds_with_fbbt`` every aux variable in these instances receives a
finite bound, so increment 1's acceptance criterion is met *without* a redundant
Python pass. This test locks that behaviour so a regression in the Rust FBBT
propagation (which would silently re-disable the lift and loosen root bounds) is
caught by CI — it is deliberately left unmarked (CI deselects ``correctness``).

Soundness note: finitizing a bound via FBBT only ever *shrinks* a variable's
feasible interval to bounds implied by the constraints, so this can never make a
relaxation unsound; the test guards that the bounds stay finite, not their exact
width.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _extract_variable_info
from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

_DATA = Path(__file__).parent / "data" / "minlplib"

# (instance, number of aux variables declared [-inf, +inf] in the raw model).
# Each instance below routes nonlinear sub-expressions (division, sqrt) through
# auxiliary variables that start unbounded; root FBBT must finitize all of them.
_AUX_CHAIN_CASES = [
    ("nvs05", 8),  # x4..x7 aux (reciprocal, sqrt, nested division) → 8 inf bounds
    ("nvs22", 8),
    ("chance", 4),
]


@pytest.mark.parametrize("instance, expected_inf_before", _AUX_CHAIN_CASES)
def test_root_fbbt_finitizes_aux_chain(instance, expected_inf_before):
    """Root FBBT eliminates every infinite aux bound (increment 1, issue #154)."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    _, lb, ub, int_offsets, int_sizes = _extract_variable_info(m)

    inf_before = int(np.sum(~np.isfinite(lb)) + np.sum(~np.isfinite(ub)))
    assert inf_before == expected_inf_before, (
        f"[{instance}] expected {expected_inf_before} infinite raw bounds, got {inf_before}"
    )

    tlb, tub, infeasible, _ = tighten_root_bounds_with_fbbt(
        m, lb.copy(), ub.copy(), int_offsets, int_sizes
    )

    # FBBT must not declare these feasible instances infeasible.
    assert not infeasible, f"[{instance}] root FBBT falsely reported infeasible"

    inf_after = int(np.sum(~np.isfinite(tlb)) + np.sum(~np.isfinite(tub)))
    assert inf_after == 0, (
        f"[{instance}] root FBBT left {inf_after} infinite aux bound(s) — "
        f"the reciprocal/sqrt lift (#157) will abstain and root bounds will loosen"
    )
    # Finitization shrinks intervals; it can never widen them (soundness).
    assert np.all(tlb >= lb - 1e-9), f"[{instance}] FBBT loosened a lower bound"
    assert np.all(tub <= ub + 1e-9), f"[{instance}] FBBT loosened an upper bound"
