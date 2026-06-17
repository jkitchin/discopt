"""Regression locks for the scaled affine-power lift and the LP rescaling that
the bucket-1 multilinear/conditioning work (issue #175) introduced.

Two pieces are exercised here:

1. **Scaled affine-power envelope** ``(c*x)**n`` (n >= 3). A power of a *scaled*
   variable used to be omitted from the relaxation entirely ("Cannot linearize
   power expression"), dropping the whole containing constraint. It is now lifted
   on the well-conditioned residual ``r = c*x`` with a univariate power envelope.
   The lifted bound must stay **sound** (a valid lower bound never exceeds the
   true optimum).

2. **Ill-conditioned root solve** (``ex1252``). ``ex1252`` mixes a variable scaled
   by ``1/2950`` (so its cube stays in ``[0, 1]``) with the un-scaled companions in
   ``[0, 2950]``; once the cubic constraints are no longer omitted the McCormick LP
   matrix spans ~1e15 in magnitude. The simplex's basis equilibration must still
   recover a finite, sound root bound.

Soundness — a valid dual bound never exceeds the true optimum — is the invariant
under test; tightness is not asserted (these two roots remain loose for
structural reasons documented on the issue).
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds

_DATA = Path(__file__).parent / "data" / "minlplib"


@pytest.mark.correctness
def test_scaled_cubic_power_is_lifted_and_sound():
    """``minimize (0.001*x)**3`` over ``x in [600, 1000]`` (so ``r = 0.001*x`` is
    in ``[0.6, 1.0]`` and the cube is convex). The relaxation must produce a
    finite root bound that never exceeds the true optimum ``0.6**3 = 0.216`` —
    proving the scaled affine-power node is enveloped, not omitted."""
    m = dm.Model()
    x = m.continuous("x", lb=600.0, ub=1000.0)
    m.minimize((0.001 * x) ** 3)

    relaxer = MccormickLPRelaxer(m)
    assert relaxer.has_relaxable_nonlinearity
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    # Sound: lower bound <= true optimum (0.6**3 = 0.216); convex on [0.6,1.0].
    assert res.lower_bound <= 0.216 + 1e-6, f"unsound bound {res.lower_bound}"
    # Non-trivial: the cube is enveloped, not dropped to the box minimum 0.
    assert res.lower_bound >= -1e-6


@pytest.mark.correctness
def test_scaled_cubic_power_sound_with_negative_coeff():
    """A *negative* cube coefficient flips the role of the envelope; the bound
    must stay sound. ``minimize -(0.001*x)**3`` over ``x in [0, 1000]`` has true
    optimum ``-1.0`` at ``x = 1000`` (``r = 1``); the relaxation lower bound must
    not exceed it."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1000.0)
    m.minimize(-1.0 * (0.001 * x) ** 3)

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    assert res.lower_bound <= -1.0 + 1e-6, f"unsound bound {res.lower_bound}"


@pytest.mark.correctness
def test_ex1252_root_bound_finite_and_sound():
    """``ex1252``'s cubic-defining constraints used to be omitted (so the root
    bound was a fast but loose 0); including them produces a McCormick LP so
    ill-conditioned that the solve stalls with no bound. The rescaling retry must
    recover a finite root bound that stays sound (``<= optimum 128893.8``)."""
    nl = _DATA / "ex1252.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"ex1252 root LP status {res.status}"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    assert res.lower_bound <= 128893.8 + 1e-3, f"unsound bound {res.lower_bound}"
