"""Regression test: nvs08 certifies via reciprocal-power canonicalization.

nvs08 (MINLPLib, optimum ≈ 23.4497) has the constraint factor
``1/(x0**3 * sqrt(x0))``. Previously the MILP relaxation could not linearize
the non-constant division, so it *dropped the whole constraint* and the solve
returned ``feasible`` with no dual bound (``bound=None``) — it found the optimum
but could not certify it.

The fix canonicalizes ``1/(x0**3 * sqrt(x0))`` to the fractional power
``x0**-3.5`` (``term_classifier.extract_reciprocal_power``) so the existing
fractional-power relaxation lifts it to an aux column. The constraint also
couples that fractional power with the integer monomial ``x2**2`` over wide
bounds — the exact shape that, before the companion soundness fix, made the
fast simplex falsely report the node LP infeasible. Both fixes together let
nvs08 certify soundly.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_NL = Path(__file__).parent / "data" / "minlplib" / "nvs08.nl"
_OPT = 23.4497


def test_nvs08_reciprocal_classified_as_fractional_power():
    """1/(x0**3*sqrt(x0)) is recognized as the fractional power x0**-3.5."""
    from discopt._jax.term_classifier import classify_nonlinear_terms

    assert _NL.exists(), f"missing {_NL}"
    terms = classify_nonlinear_terms(dm.from_nl(str(_NL)))
    assert (0, -3.5) in terms.fractional_power


@pytest.mark.correctness
def test_nvs08_certifies_to_optimum():
    """nvs08 certifies its MINLPLib optimum with a valid dual bound."""
    r = dm.from_nl(str(_NL)).solve(time_limit=120, gap_tolerance=1e-4)
    assert r.status == "optimal", f"status={r.status} (was 'feasible' before the fix)"
    assert r.objective is not None
    assert abs(r.objective - _OPT) <= 1e-2, f"obj={r.objective} != {_OPT}"
    # Soundness invariant: a valid dual bound never exceeds the optimum.
    assert r.bound is not None, "certified solve must report a dual bound"
    assert r.bound <= r.objective + 1e-3, f"invalid dual bound {r.bound} > obj {r.objective}"
