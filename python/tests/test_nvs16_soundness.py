"""Regression test for issue #120: invalid McCormick "nlp" dual bound.

On a nonconvex pure-integer NLP, the McCormick ``"nlp"`` lower-bounding path
used to certify a *local* objective value as a global lower bound. The bound
solver evaluates the compiled relaxation at ``x_cv == x_cc``, where every
McCormick rule is tight, so it minimizes the original (nonconvex) objective
locally — a value that can lie ABOVE the true optimum. For MINLPLib ``nvs16``
this produced ``status=optimal, obj=bound=14.203125, nodes=1`` while the true
optimum is ``0.703125`` (a silent false-optimal).

The fix gates the ``"nlp"`` bound on convexity: nonconvex models fall back to
the rigorous alphaBB underestimator. ``nvs16`` must now converge to the true
optimum, and no bounding mode may emit a dual bound above it.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_NL = Path(__file__).parent / "data" / "minlplib" / "nvs16.nl"

# Global optimum from MINLPLib (SCIP and Couenne agree).
_OPT = 0.703125


@pytest.mark.correctness
def test_nvs16_default_is_sound():
    """Default solve must not certify a bound above the true optimum."""
    assert _NL.exists(), f"missing {_NL}"
    m = dm.from_nl(str(_NL))
    r = m.solve(time_limit=60)

    assert r.objective is not None
    # Primal must reach the true optimum (not the spurious local 14.203125).
    assert abs(r.objective - _OPT) <= 1e-3, f"obj={r.objective} != {_OPT}"
    # Soundness invariant: a valid dual bound never exceeds the optimum.
    if r.bound is not None:
        assert r.bound <= _OPT + 1e-3, f"invalid dual bound {r.bound} > optimum {_OPT} (issue #120)"


@pytest.mark.correctness
@pytest.mark.parametrize("mode", ["auto", "none", "nlp", "lp"])
def test_nvs16_all_bounding_modes_sound(mode):
    """No McCormick bounding mode may certify an invalid bound on nvs16.

    ``"nlp"`` and the pure-integer ``"lp"``→``"nlp"`` fallback are downgraded
    to the alphaBB underestimator for this nonconvex model; all four modes must
    return the true optimum with a valid (``<= optimum``) dual bound.
    """
    m = dm.from_nl(str(_NL))
    r = m.solve(time_limit=60, mccormick_bounds=mode)

    assert r.objective is not None
    assert abs(r.objective - _OPT) <= 1e-3, f"[{mode}] obj={r.objective}"
    if r.bound is not None:
        assert r.bound <= _OPT + 1e-3, f"[{mode}] invalid dual bound {r.bound} > optimum {_OPT}"
