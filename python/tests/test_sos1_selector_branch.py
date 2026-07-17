"""SOS1 continuous-selector spatial branching (issue #196).

``ex1252``'s global dual bound is pinned at 0 because its objective is a sum of
products gated by line indicators ``x18/x19/x20``; on a box where several lines
are simultaneously "on" the *continuous* selectors ``x21+x22+x23 = 1`` (each
``x21 <= x18`` ...) stay spread and the McCormick relaxation drives every product
to 0. Branching a selector spatially concentrates the selection so a single
product is forced positive (measured: an ambiguous box's bound 12658 -> ~67-83k
once one selector is pinned).

The mechanism is gated behind ``DISCOPT_SOS1_SELECTOR_BRANCH`` (default OFF) and
is **branch-ORDER metadata only** — it never enters a bound or feasibility test,
so these tests lock (a) that the detector finds exactly the selectors, (b) that
it does not fire on unrelated structure, and (c) that enabling it keeps every
reported bound *sound* (<= the known optimum). Soundness is the invariant; the
bound-lift is measured in the issue, not asserted as a brittle threshold here.
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"
_EX1252_OPT = 128893.74


@pytest.mark.correctness
def test_ex1252_sos1_selector_detection():
    """The detector identifies exactly ex1252's continuous selectors
    ``x21/x22/x23`` — the members of the selection row ``x21+x22+x23=1``, each
    upper-coupled to a ``[0,1]`` indicator (``x21<=x18`` ...) whose indicator gates
    an objective product. Branch-ORDER metadata only, so this guards the
    heuristic, not soundness."""
    from discopt.solver import _sos1_selector_vars

    nl = _DATA / "ex1252.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))
    sel = _sos1_selector_vars(m)
    assert sorted(sel) == [21, 22, 23], f"expected selectors {{21,22,23}}, got {sorted(sel)}"


@pytest.mark.correctness
def test_sos1_selector_no_false_positive_plain_bilinear():
    """A plain bilinear model with no one-of-N selection row yields no selectors:
    the detector must not fire on generic products."""
    from discopt.solver import _sos1_selector_vars

    m = dm.Model("plain_bilinear")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x * y)
    m.subject_to(x + y >= 0.5)  # inequality, not a ``= const`` selection row
    assert _sos1_selector_vars(m) == frozenset()


@pytest.mark.correctness
def test_sos1_selector_no_false_positive_linear_selection():
    """A one-of-N ``= 1`` selection row whose members are NOT coupled to a
    product-gating indicator (a purely linear assignment) yields no selectors:
    concentrating them would not tighten any nonlinear relaxation."""
    from discopt.solver import _sos1_selector_vars

    m = dm.Model("linear_selection")
    s1 = m.continuous("s1", lb=0.0, ub=1.0)
    s2 = m.continuous("s2", lb=0.0, ub=1.0)
    m.minimize(s1 + 2.0 * s2)  # linear objective — no product gated by an indicator
    m.subject_to(s1 + s2 == 1.0)
    assert _sos1_selector_vars(m) == frozenset()


@pytest.mark.correctness
def test_ex1252_sos1_selector_branch_is_sound():
    """Enabling the selector spatial branch keeps every reported bound sound:
    the dual bound never exceeds the true optimum and a certified gap never
    certifies a bound above it. Order-only metadata cannot loosen a bound, so the
    lock is that the *lifted* bound is still valid."""
    monkey = os.environ.get("DISCOPT_SOS1_SELECTOR_BRANCH")
    os.environ["DISCOPT_SOS1_SELECTOR_BRANCH"] = "1"
    try:
        nl = _DATA / "ex1252.nl"
        assert nl.exists(), f"missing {nl}"
        m = dm.from_nl(str(nl))
        r = m.solve(time_limit=20.0)
        # Dual bound must be a valid lower bound: <= the true optimum, always.
        if r.bound is not None and math.isfinite(r.bound):
            assert r.bound <= _EX1252_OPT + 1e-3, (
                f"ex1252 SOS1-branch UNSOUND dual bound {r.bound} > optimum {_EX1252_OPT}"
            )
        # Any incumbent found must be feasible (>= optimum for this minimize).
        if r.objective is not None and math.isfinite(r.objective):
            assert r.objective >= _EX1252_OPT - 1e-3, (
                f"ex1252 SOS1-branch incumbent {r.objective} below optimum {_EX1252_OPT}"
            )
        # A certified gap must bracket the optimum soundly.
        if r.gap_certified and r.bound is not None and r.objective is not None:
            assert r.bound <= r.objective + 1e-3
    finally:
        if monkey is None:
            os.environ.pop("DISCOPT_SOS1_SELECTOR_BRANCH", None)
        else:
            os.environ["DISCOPT_SOS1_SELECTOR_BRANCH"] = monkey
