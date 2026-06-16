"""End-to-end locks for root-relaxation bound surfacing on the #138 bucket.

The spatial ``solve_model`` path previously dropped its tree bound on an
uncertified exit and reported ``bound=None`` for the hard bucket-1 instances. It
now falls back to a rigorous root MILP-relaxation bound over the root box (the
MILP path already did this; this brings the spatial path to parity), combined
with the fractional-power-of-product objective lift and a pre-reform FBBT pass.
The result: the whole bucket reports a finite, *sound* dual bound.

Soundness is the bar — a dual bound for a minimization must never exceed the true
optimum. These pin both that the bound is now finite (no regression to ``None``)
and that it stays sound.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"

# MINLPLib optima (minlplib.solu): a valid dual lower bound can never exceed these.
_BUCKET = {
    "gear4": 1.64342847,  # trilinear ratio, integrality gap; floor 0 from x4,x5>=0
    "nvs22": 6.0581153,  # var/var ratio
    "st_e35": 64868.6,  # (x / g**(1/3))**0.83 fractional-power-of-product
    "ex1233": 155010.6713,  # x / g**(1/3); geometric vars finitized by pre-reform FBBT
}


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", sorted(_BUCKET.items()))
def test_bucket_returns_finite_sound_bound(instance, optimum):
    """Each hard bucket-1 instance now reports a finite, sound dual bound."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=40, gap_tolerance=1e-4)

    assert r.bound is not None, f"[{instance}] regressed to bound=None"
    # Soundness: the dual bound never exceeds the known optimum.
    assert r.bound <= optimum + max(1e-2, abs(optimum) * 1e-3), (
        f"[{instance}] unsound dual bound {r.bound} > optimum {optimum}"
    )
    # And never certifies optimality without an incumbent.
    assert not (r.gap_certified and r.objective is None), (
        f"[{instance}] certified with no incumbent"
    )
