"""Bucket-2 (#139) regression: every "product with a nonlinear factor" instance
produces a *sound* root lower bound — or, for the one slice that legitimately
drops, does not regress relative to a known-good baseline.

Bucket 2 covers products where at least one factor is itself nonlinear (a square,
a product, or another aux expression): ``ex1225``, ``ex1226``, ``ex1252``,
``nvs05``, ``nvs16``, ``nvs20``, ``nvs22``, ``chance``, ``st_e36``. Before #139
each of these dropped from the McCormick relaxation ("Cannot decompose product")
and produced no dual bound; recursive bilinear/trilinear/multilinear lifting plus
the extreme-magnitude monomial guard now lift them soundly.

Soundness is the invariant under test: a valid lower bound must NEVER exceed the
true optimum. We assert the root McCormick LP bound is finite and ``<= opt`` for
the eight liftable instances. ``nvs16`` is the lone exception — its objective is
a Beale sum-of-squares that distributes into degree-8 monomials of magnitude
~1e18 over the integer box ``[0, 200]**2``; a proper square-of-affine-in-lifted
envelope is follow-up work. It currently yields no root objective bound on *main*
too, so we assert only that it has not regressed (no bound, or a sound one).

Reference optima are the MINLPLib values (cross-checked against
``discopt_benchmarks`` problem definitions).
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

# (instance, MINLPLib optimum). These eight must each produce a finite, sound
# root lower bound (bound <= opt). Bounds may be loose where division/sqrt
# constraints remain un-linearized — looseness is fine, unsoundness is not.
_SOUND_BOUND_CASES = [
    ("ex1225", 31.0),
    ("ex1226", -17.0),
    ("ex1252", 128893.8),
    ("nvs05", 5.47093),
    ("nvs20", 230.922),
    ("nvs22", 6.0584),
    ("chance", 29.8945),
    ("st_e36", -246.0),
]


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", _SOUND_BOUND_CASES)
def test_bucket2_instance_has_sound_root_bound(instance, optimum):
    """Each liftable bucket-2 instance yields a finite root bound <= optimum."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"[{instance}] root LP status {res.status}"
    assert res.lower_bound is not None, f"[{instance}] objective dropped — no root bound"
    assert math.isfinite(res.lower_bound), f"[{instance}] non-finite bound {res.lower_bound}"
    # The soundness invariant: a valid lower bound never exceeds the true optimum.
    assert res.lower_bound <= optimum + 1e-3, (
        f"[{instance}] UNSOUND root bound {res.lower_bound} > optimum {optimum}"
    )


@pytest.mark.correctness
def test_nvs16_does_not_regress():
    """``nvs16`` (Beale sum-of-squares, integer box [0,200]^2) distributes into
    ~1e18-magnitude monomials, so its objective legitimately drops from the root
    relaxation today. The bucket-2 acceptance requires only that it not regress:
    the root LP must still solve (no spurious infeasibility), and any bound it
    does produce must be sound (<= optimum 0.703125)."""
    nl = _DATA / "nvs16.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    # No false-infeasibility — the LP itself must solve cleanly.
    assert res.status == "optimal", f"nvs16 root LP status {res.status}"
    # The objective may drop (no bound) — that is the current, non-regressed
    # state — but if a bound IS produced it must underestimate the optimum.
    if res.lower_bound is not None and math.isfinite(res.lower_bound):
        assert res.lower_bound <= 0.703125 + 1e-3, (
            f"nvs16 UNSOUND bound {res.lower_bound} > 0.703125"
        )
