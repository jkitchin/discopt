"""Regression locks for variable/variable ratio (quotient) certification (issue #138, bucket 1).

A constraint term ``N / D`` with a non-constant, sign-definite denominator ``D``
is cleared by multiplying the constraint through by ``D`` (exact and
value-preserving over a box where ``D`` never vanishes; see
``factorable_reform._clear_divisions``), exposing a polynomial term the McCormick
relaxation can bound. Combined with the mixed repeated-factor product lift this
already certifies a slice of bucket-1 instances **soundly** on ``main``:

* the gear-train problems ``gear`` / ``gear2`` / ``gear3`` — bilinear/bilinear
  ratio ``(x0*x1)/(x2*x3)`` whose continuous relaxation floor is 0 and whose
  incumbent reaches ~0;
* ``nvs01`` (12.46967) and ``nvs06`` (1.7703125) — quadratic/quadratic ratios.

These tests pin that behavior. ``st_e17`` (``x0**2/x1`` via convex-clearing) is
guarded separately by ``test_st_e17_certification.py``.

Scope note (issue #138): the remaining bucket-1 instances ``gear4``, ``ex1233``,
``nvs05``, ``nvs22``, ``st_e35`` currently return ``bound=None`` (the relaxation
soundly *drops* the ratio rather than producing a false bound) and are **not**
locked here. The naive "lift ``y = N/D`` to the bilinear equality ``y*D = N``"
route is deliberately NOT taken: on ``gear4`` it produces a deterministic
false-optimal (the large linking coefficient drives an invalid node lower bound),
which would convert a sound ``bound=None`` into an unsound certified-optimal. That
is tracked as a separate solver soundness bug (#145); closing bucket-1 soundly
waits on it plus tighter division-specific envelopes.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"


@pytest.mark.correctness
@pytest.mark.parametrize("instance", ["gear", "gear2", "gear3"])
def test_gear_ratio_certifies_near_zero(instance):
    """Gear-train ``(x0*x1)/(x2*x3)`` ratios certify to ~0 with a sound bound."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=60, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None, f"[{instance}] no bound produced"
    # Global optimum of these gear problems is ~0; the incumbent reaches it.
    assert r.objective <= 1e-3, f"[{instance}] obj={r.objective} not near 0"
    # Soundness: a valid dual bound never exceeds the optimum.
    assert r.bound <= r.objective + 1e-6, f"[{instance}] unsound bound {r.bound} > obj"
    assert r.gap_certified, f"[{instance}] expected certified optimality"


@pytest.mark.correctness
@pytest.mark.parametrize(
    "instance, optimum",
    [
        ("nvs01", 12.46966882),  # (18505*x1**2 + ...) / (x0**2 + 7200) quadratic/quadratic
        ("nvs06", 1.7703125),  # variable/variable ratio
    ],
)
def test_nvs_ratio_certifies(instance, optimum):
    """nvs01 / nvs06 variable/variable ratios certify their optimum soundly."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=60, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None, f"[{instance}] no bound produced"
    assert abs(r.objective - optimum) <= 1e-2, f"[{instance}] obj={r.objective} != {optimum}"
    # Soundness: a valid dual bound never exceeds the known global optimum.
    assert r.bound <= optimum + 1e-2, f"[{instance}] unsound dual bound {r.bound} > {optimum}"
    assert r.gap_certified, f"[{instance}] expected certified optimality"
