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
# true optimum, so a sound bound must always sit at or below these.
#
# NOTE on ex1233: the value is 155010.6713, NOT the 62.1833 long recorded in
# ``discopt_benchmarks`` KNOWN_OPTIMA. That figure is demonstrably wrong — the
# spatial relaxation now produces a *rigorous* (``gap_certified``) lower bound of
# ~109649 on ex1233, and the solver finds a feasible incumbent at 155010.671
# (the MINLPLib global optimum). A valid lower bound of 109649 means no feasible
# point has objective below it, so 62.1833 cannot be the optimum. The benchmark
# value has been corrected alongside this change.
_BUCKET1_WEAK = {
    "gear4": 1.64342847,  # (x0*x1)/(x2*x3) equality, integer — pure integrality gap
    "ex1233": 155010.6713,  # linear / (product)^(1/3); corrected from a wrong 62.1833
    "nvs05": 5.47093411,  # var/var ratio + sqrt-of-product in defining equalities
    "nvs22": 6.0581153,  # var/var ratio
    "st_e35": 64868.6,  # (x / (product)^(1/3))^0.83 in the objective
}

# Subset that now returns a *finite* sound bound (issue #138):
#   nvs05 / nvs22 — via the root MILP-relaxation fallback over the sanitized
#     relaxation (their cleared-division equalities over the wide-ranged defined
#     variables x4..x7 would otherwise leave the LP unsolvable);
#   ex1233 / st_e35 — via the fractional-power-of-product objective lift: each
#     term ``(x / g**(1/3))**0.83`` (st_e35) or ``x / g**(1/3)`` (ex1233) is
#     decomposed into elementary aux variables (g -> t, t**(1/3) -> d, x/d -> r,
#     r**0.83 -> s) so the objective becomes linear in the aux and the relaxation
#     can bound it. ex1233 additionally needs its geometric variables bounded,
#     which a pre-reform FBBT pass now supplies (they are declared unbounded but
#     pinned finitely by the assignment constraints).
_BUCKET1_NOW_BOUNDED = {
    "ex1233": 155010.6713,
    "nvs05": 5.47093411,
    "nvs22": 6.0581153,
    "st_e35": 64868.6,
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
def test_bucket1_instance_returns_finite_sound_bound(instance, optimum):
    """ex1233 / nvs05 / nvs22 / st_e35 now return a *finite* sound lower bound.

    nvs05 / nvs22: clean polynomial objective lifted to bilinear form, but the
    spatial tree bound is uncertified and dropped and the relaxation's
    cleared-division equalities over the wide-ranged defined variables x4..x7
    produce envelope entries up to ~1e37 that leave the LP unsolvable; the
    root-relaxation fallback sanitizes those catastrophic rows/bounds (sound,
    since dropping a constraint or widening a box only relaxes) and solves it.

    ex1233 / st_e35: the fractional-power-of-product objective term is decomposed
    into elementary aux variables so the objective becomes linear in the aux and
    the relaxation can bound it; ex1233 additionally relies on a pre-reform FBBT
    pass to bound its (declared-unbounded) geometric variables. ex1233's bound is
    rigorous and ~109649 — far above the 62.1833 once wrongly recorded as its
    optimum, confirming that value was below a valid lower bound.

    In every case the dual bound stays at or below the true optimum (never false).
    """
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=40, gap_tolerance=1e-4)

    assert r.bound is not None, f"[{instance}] produced no finite bound (regressed to None)"
    # Soundness: a valid dual lower bound never exceeds the known optimum.
    assert r.bound <= optimum + 1e-2, f"[{instance}] unsound dual bound {r.bound} > {optimum}"


@pytest.mark.parametrize(
    "build, expected",
    [
        # ex1233-shaped term  N / g**(1/3)  with bounded positive vars.
        ("ex1233_ratio", 60.00966),
        # st_e35-shaped term  (N / g**(1/3))**0.83  with bounded positive vars.
        ("st_e35_power", 176.16932),
    ],
)
def test_fractional_power_of_product_envelope_is_sound(build, expected):
    """The fractional-power-of-product objective lift certifies the known optimum
    of a small, fully-bounded instance (issue #138).

    These pin the envelope itself (independent of the slow MINLPLib ``.nl``
    solves): the automatic ``factorable_reform`` decomposition of
    ``N / g**p`` and ``(N / g**p)**q`` must reproduce the value computed by hand
    and certify a sound bound (``bound <= objective``).
    """
    m = dm.Model(build)
    x = m.continuous("x", lb=1.0, ub=10.0)
    xa = m.continuous("xa", lb=1.0, ub=5.0)
    xb = m.continuous("xb", lb=1.0, ub=5.0)
    if build == "ex1233_ratio":
        m.minimize(300 * x / (0.5 * (xa**2 * xb + xa * xb**2)) ** 0.3333)
    else:
        m.minimize(670 * (x / (0.5 * (xa**2 * xb + xb**2 * xa)) ** 0.333333) ** 0.83)

    r = m.solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None and r.bound is not None
    assert abs(r.objective - expected) <= 1e-1, f"obj {r.objective} != expected {expected}"
    # Soundness: the dual bound never exceeds the objective at the optimum.
    assert r.bound <= r.objective + 1e-4, f"unsound bound {r.bound} > obj {r.objective}"
    assert r.gap_certified, "small fully-bounded instance should certify"
