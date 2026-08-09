"""Issue #971: the root-LP gate must model *both* ways a difference can arise.

``diff_root_lp`` (#961) excused a root-LP difference from being a claim change
only when the relaxation *fingerprint* differed too, reasoning that "a difference
is only attributable when both sides solved the same LP". That rule assumes
**same matrix bytes ⇒ same LP answer**, and CI falsified it: ``contvar`` was
bucketed ``changed`` — the fingerprint MATCHED — with the root LP bound moving
from ``172170.3107274997`` to ``187283.4711213944`` (8.8%) between two
ubuntu-x86-64 runners, bit-identically on each. The gate hard-failed on ``main``
and on PRs touching no solver code.

The mechanism is not last-digit noise. The certified bound is the
Neumaier-Shcherbina safe bound built from the simplex's own dual vertex, and a
degenerate LP's optimal dual is not unique: an arithmetic-path difference that
flips one ratio-test tie-break selects a different — equally valid — dual, and
the bound it certifies moves discretely.
``discopt_benchmarks/scripts/issue971_root_lp_arithmetic_path.py`` reproduces
the mechanism on demand, holding tree/instance/fingerprint fixed while varying
only the OpenBLAS kernel: 5 kernels, 1 fingerprint, 3 distinct bounds.

The fix is not a tolerance bump (8.8% is not noise, and widening a correctness
gate to turn a red green is what CLAUDE.md §3 forbids). It splits the gate into
the part that holds on every host — the #961 status/bound contract and the
bound's validity against the published optimum, asserted hard — and the part that
does not, the exact float, which is reported and bounded.
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from support.claim_differential import (
    MAX_UNREPRODUCED_ROOT_LP,
    RootLpProbe,
    current_row,
    diff_root_lp,
    load_baseline,
    reference_optima,
    root_lp_violations,
)

pytestmark = [pytest.mark.claim_boundary]


# ── the host-independent property predicate ────────────────────────────────────
# Pure-function tests: no solve, so they run in the fast suite and prove the
# predicate bites on every arm (CLAUDE.md §6 — an instrument that cannot fail is
# not an instrument).


def test_violations_flag_the_961_contract_in_both_directions():
    assert root_lp_violations(RootLpProbe("x", "optimal", None, False))
    assert root_lp_violations(RootLpProbe("x", "optimal", float("nan"), False))
    assert root_lp_violations(RootLpProbe("x", "optimal", float("inf"), False))
    # A non-optimal status may not carry a bound either.
    assert root_lp_violations(RootLpProbe("x", "uncertified", 1.0, False))
    # ...and the admissible pairs are admissible.
    assert not root_lp_violations(RootLpProbe("x", "optimal", 1.0, False))
    assert not root_lp_violations(RootLpProbe("x", "uncertified", None, False))


def test_violations_flag_a_bound_above_the_reference_optimum():
    name, optimum = next(iter(sorted(reference_optima().items())))
    # A rigorous lower bound may sit anywhere below the optimum, however loose.
    assert not root_lp_violations(RootLpProbe(name, "optimal", optimum - 1e6, False))
    # It may not sit above it.
    above = optimum + 1.0 + 1e-2 * abs(optimum)
    assert root_lp_violations(RootLpProbe(name, "optimal", above, False)), (
        f"a bound of {above} above {name}'s optimum {optimum} was not flagged"
    )
    # An unknown instance has no oracle, so validity is simply not asserted --
    # which is exactly why the gate floors how many instances WERE checked.
    assert not root_lp_violations(RootLpProbe("no-such-instance", "optimal", above, False))


def test_violations_respect_the_objective_sense():
    """For a MAXIMIZE model the internal objective is negated, so the internal
    lower bound must not exceed ``-optimum``; comparing it to ``+optimum`` would
    both miss real violations and manufacture false ones."""
    name, optimum = next(iter(sorted(reference_optima().items())))
    # Internal floor is -optimum: a bound just under it is admissible...
    assert not root_lp_violations(RootLpProbe(name, "optimal", -optimum - 1.0, True))
    # ...and one clearly above it is not.
    assert root_lp_violations(
        RootLpProbe(name, "optimal", -optimum + 1.0 + 1e-2 * abs(optimum), True)
    )


# ── the attributability rule ──────────────────────────────────────────────────


@pytest.mark.slow
def test_same_bytes_different_bound_is_not_a_claim_change():
    """The #971 regression: identical fingerprint, different bound.

    Before the fix this returned ``changed`` and hard-failed the gate — which is
    exactly what CI did to ``contvar`` on runs that touched no solver code. It
    must land in the bounded, informational ``unreproduced`` bucket instead.
    """
    baseline = load_baseline()
    name = next(
        (n for n in sorted(baseline) if diff_root_lp(n, baseline).status == "unchanged"),
        None,
    )
    assert name is not None, "no exactly-reproduced instance to test the rule with"
    row = dict(baseline[name])
    assert row["fingerprint"] == current_row(name)["fingerprint"], "bytes must match to test this"

    # Reproduce contvar's shape: same recorded bytes, a bound 8.8% away.
    base_bound = row["root_lp_bound"]
    moved = 187283.4711213944 if base_bound is None else base_bound * 1.088 + 1.0
    doctored = dict(row, root_lp_status="optimal", root_lp_bound=moved)
    d = diff_root_lp(name, {name: doctored})
    assert d.status == "unreproduced", f"same-bytes bound difference misfiled on {name}: {d}"
    assert "identical matrix bytes" in d.detail


def test_the_escape_hatch_is_bounded():
    """An excused bucket that can grow without limit is a gate that stopped
    measuring. ``contvar`` is the only instance ever observed to do this, so the
    cap stays within a couple of instances of that evidence."""
    assert 1 <= MAX_UNREPRODUCED_ROOT_LP <= 3
