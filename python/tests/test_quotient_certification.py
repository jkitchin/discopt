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
``nvs05``, ``nvs22``, ``st_e35`` return ``bound=None`` (the relaxation soundly
*drops* the ratio rather than producing a false bound). The naive "lift
``y = N/D`` to the bilinear equality ``y*D = N``" route is deliberately NOT taken:
on ``gear4`` it produced a deterministic false-optimal (the large linking
coefficient drove an invalid node lower bound), which would have converted a sound
``bound=None`` into an unsound certified-optimal. That was tracked and fixed as a
separate solver soundness bug (#145, now closed via an exact LP oracle for OBBT);
closing bucket-1 with an *actual* finite bound additionally needs the
division-specific / fractional-power-of-product envelopes shared with buckets #2
and #4.

These five are now regression-locked here for **soundness** (not for a certified
optimum): each must return a sound result — never a dual bound above the known
optimum, and never ``gap_certified=True`` without an incumbent. The latter guards
a spurious-certification bug uncovered while picking up #138: a resource-limit
termination with no incumbent left the ``SolveResult.gap_certified`` default of
``True`` in place (together with a phantom near-zero bound and ``gap=inf``), so
``ex1233`` reported ``gap_certified=True`` with ``objective=None``. The no-incumbent
branches of the B&B finalizers now clear ``gap_certified`` for resource-limit
exits (an exhausted-tree ``infeasible`` conclusion stays certified).
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


# Known global optima (MINLPLib) used as an upper reference for the soundness
# guard below: a valid dual lower bound for a minimization can never exceed the
# true optimum. None of these *certify* on ``main`` (the gap stays open), but
# they must never report a false bound. ``gear4`` (separable floor) and
# ``nvs05`` / ``nvs22`` (root MILP-relaxation fallback) now additionally return a
# finite, weak-but-sound bound; ``ex1233`` / ``st_e35`` still drop their
# fractional-power-of-product objective term entirely and stay ``bound=None``.
_BUCKET1_WEAK = {
    "gear4": 1.64342847,  # (x0*x1)/(x2*x3) equality, integer — pure integrality gap
    "ex1233": 62.1833,  # linear / (product)^(1/3) plus sqrt-of-product terms
    "nvs05": 5.47093411,  # var/var ratio + sqrt-of-product in defining equalities
    "nvs22": 6.0581153,  # var/var ratio
    "st_e35": 64868.6,  # (x / (product)^(1/3))^0.83 in the objective
}

# Subset that now returns a *finite* sound bound via the root MILP-relaxation
# fallback (issue #138): the objective lifts cleanly (factorable_reform), and the
# fallback solves the sanitized root relaxation whose catastrophically scaled
# rows/bounds — from cleared-division equalities over the wide-ranged defined
# variables x4..x7 — would otherwise leave the LP unsolvable.
_BUCKET1_NOW_BOUNDED = {
    "nvs05": 5.47093411,
    "nvs22": 6.0581153,
}


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", sorted(_BUCKET1_WEAK.items()))
def test_bucket1_weak_instances_stay_sound(instance, optimum):
    """The not-yet-certified bucket-1 instances must stay **sound**.

    Two invariants, both of which a previous build violated on ``ex1233``:

    1. A presented dual bound never exceeds the known optimum (no false bound).
    2. ``gap_certified=True`` is never reported without an incumbent — a
       resource-limit termination that found no feasible solution must not claim
       optimality just because the ``SolveResult.gap_certified`` field defaults to
       ``True`` (issue #138 follow-up; the no-incumbent B&B exit now clears it).
    """
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=30, gap_tolerance=1e-4)

    # (1) No false dual bound for a minimization: bound <= true optimum.
    if r.bound is not None:
        assert r.bound <= optimum + 1e-2, f"[{instance}] unsound dual bound {r.bound} > {optimum}"

    # (2) Certification requires an incumbent; if certified, it must be genuine.
    if r.gap_certified:
        assert r.objective is not None, (
            f"[{instance}] gap_certified=True with no incumbent (objective is None)"
        )
        assert r.bound is not None and r.bound <= r.objective + 1e-6, (
            f"[{instance}] certified but bound {r.bound} > objective {r.objective}"
        )


@pytest.mark.correctness
def test_ex1233_no_spurious_certification():
    """ex1233 hits the time limit with no incumbent; it must report an honest,
    uncertified result — regression for the spurious ``gap_certified=True`` /
    phantom near-zero bound / ``gap=inf`` returned when a no-incumbent
    resource-limit exit left the ``gap_certified`` default of ``True`` in place.
    """
    nl = _DATA / "ex1233.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=20, gap_tolerance=1e-4)

    # The fundamental invariant: never certify optimality without an incumbent.
    assert not (r.gap_certified and r.objective is None), (
        f"ex1233 certified with no incumbent: status={r.status} bound={r.bound} gap={r.gap}"
    )


@pytest.mark.correctness
def test_gear4_returns_finite_sound_bound():
    """gear4 returns a *finite* sound lower bound, not ``None`` (issue #138).

    gear4 minimizes ``x4 + x5`` (both >= 0) subject to a single trilinear-ratio
    equality with integer ``x0..x3``; the whole optimality gap is integrality, so
    the continuous relaxation floor is provably 0 and the spatial tree bound is
    uncertified (and dropped). The solver now falls back to the rigorous separable
    objective floor over the root box — here ``min(x4 + x5) = 0`` from ``x4,x5 >=
    0`` — so a finite, sound dual bound is reported instead of ``None``. The true
    optimum is ~1.6434, so ``0 <= opt`` holds: weak but never false.
    """
    nl = _DATA / "gear4.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=30, gap_tolerance=1e-4)

    assert r.bound is not None, "gear4 produced no finite bound (regressed to None)"
    # Sound: a valid lower bound never exceeds the known optimum (~1.6434).
    assert r.bound <= 1.64342847 + 1e-6, f"gear4 unsound bound {r.bound} > optimum"
    # The trivial-but-rigorous floor from x4,x5 >= 0 is 0.
    assert r.bound >= -1e-9, f"gear4 bound {r.bound} below the x4,x5>=0 floor"


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", sorted(_BUCKET1_NOW_BOUNDED.items()))
def test_nvs_ratio_returns_finite_sound_bound(instance, optimum):
    """nvs05 / nvs22 now return a *finite* sound lower bound, not ``None``.

    Their objective is a clean polynomial (e.g. ``1.10471*x0**2*x1 + ...``) that
    ``factorable_reform`` lifts to bilinear form, but the spatial tree bound is
    uncertified and dropped, and the AMP MILP relaxation's cleared-division
    equalities (``x4*x0*x1``, ``x5*x0*_fr_aux_0``) over the wide-ranged defined
    variables x4..x7 produce envelope entries up to ~1e37 that leave the LP
    unsolvable. The root-relaxation fallback (issue #138) sanitizes those
    catastrophic rows/bounds — sound, since dropping a constraint or widening a
    box only relaxes — and solves the conditioned relaxation, yielding a finite
    dual bound well below the optimum (weak, never false).
    """
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=30, gap_tolerance=1e-4)

    assert r.bound is not None, f"[{instance}] produced no finite bound (regressed to None)"
    # Soundness: a valid dual lower bound never exceeds the known optimum.
    assert r.bound <= optimum + 1e-2, f"[{instance}] unsound dual bound {r.bound} > {optimum}"
