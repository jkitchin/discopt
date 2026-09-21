"""No solve path may report a sentinel-derived value as a dual bound (#1401).

The LP layer's "no bound" is the sentinel ``1e20``, **not** ``inf`` (CLAUDE.md:
"``INF`` in the Rust LP layer is the sentinel ``1e20``, not ``f64::INFINITY``").
It survives arithmetic as an ordinary finite number, so a gate written as
``np.isfinite(v)`` accepts it and reports "the relaxation proved nothing" as a
proved bound. ``_finalize_reported_bound`` refuses such a value at ``1e19``; the
spatial B&B exit took ``stats["global_lower_bound"]`` behind ``np.isfinite``
alone and never called the composer.

Measured before the fix on ``QPLIB_2967`` (MAXIMIZE, ``time_limit=20``):
``bound=4.4495549999999977e+21``, ``gap=4.07e20``, ``gap_certified=False``.

What is pinned here is the *class*, over a sweep of box magnitudes, plus the
two-constants confusion that made the first attempt at the fix a no-op: the
effective-infinity threshold is ``1e19`` and is **not** ``SENTINEL_THRESHOLD``
(``1e29``, the infeasibility-marker / bogus-incumbent threshold). Nothing here
depends on ``QPLIB_2967``; the corpus row lives in
``scripts/audit_1401_sentinel_bound_reported.py``.
"""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest
from discopt import modeling as dm
from discopt import solver as sv

#: One order below ``CONSTRAINT_INF``; see ``solver._EFF_INF_BOUND``.
EFF_INF = 1e19

#: Box half-widths whose McCormick corner value ``b**2`` spans the sentinel from
#: two orders below it (1e16) to two above (1e24).
BOXES = (1e8, 1e9, 1e10, 1e11, 1e12)


def _sentinel_box_model(bound: float, maximize: bool):
    """A bilinear term over ``[0, bound]^2`` -- corner value ``bound**2``.

    At ``bound=1e10`` the relaxation carries ``1e20`` with no user-visible
    infinity anywhere in the model, which is the point: the sentinel is reached
    by ordinary arithmetic on ordinary declared bounds.
    """
    m = dm.Model(f"s1401_{'max' if maximize else 'min'}_{bound:g}")
    x = m.continuous("x", lb=0.0, ub=bound)
    y = m.continuous("y", lb=0.0, ub=bound)
    z = m.continuous("z", lb=0.0, ub=bound)
    m.subject_to(x * y - z <= 1.0)
    m.subject_to(x + y >= 1.0)
    if maximize:
        m.maximize(z - x)
    else:
        m.minimize(x - z)
    return m


# --------------------------------------------------------------------------
# The predicate itself: the two-constants confusion, pinned.
# --------------------------------------------------------------------------


def test_the_effective_infinity_threshold_is_not_the_infeasibility_marker():
    """The defect that made the first fix a no-op.

    ``solver._SENTINEL_THRESHOLD`` is ``1e29`` -- it answers "is this an
    infeasibility marker / bogus incumbent?", sitting just below
    ``INFEASIBILITY_SENTINEL = 1e30``. Reusing it as the effective-infinity gate
    waves every 1e20-derived bound straight through. The two constants must stay
    distinct and ordered.
    """
    assert sv._EFF_INF_BOUND == EFF_INF
    assert sv._SENTINEL_THRESHOLD == 1e29
    assert sv._EFF_INF_BOUND < sv._SENTINEL_THRESHOLD, (
        "the effective-infinity refusal must be STRICTLY below the "
        "infeasibility-marker threshold, or it refuses nothing the marker does not"
    )


def test_the_composer_and_the_predicate_share_one_constant():
    """§3, no drift: two literals for one threshold is how they diverge.

    ``_finalize_reported_bound`` used to carry its own ``_eff_inf = 1e19``. If a
    future edit reintroduces a bare literal, the two gates can drift silently and
    this test is the only thing that notices.
    """
    src = inspect.getsource(sv._finalize_reported_bound)
    assert "_EFF_INF_BOUND" in src, (
        "the composer no longer references the shared constant -- a private literal has reappeared"
    )
    # The docstring and comments legitimately *discuss* 1e19 and 1e29 -- that prose is
    # exactly what the first attempt at #1401 misread -- so look for an assignment of
    # a bare literal, not a mention of one.
    reassigned = [
        ln
        for ln in src.splitlines()
        if re.search(r"=\s*1e19\b", ln) and not ln.lstrip().startswith("#")
    ]
    assert not reassigned, (
        "a bare 1e19 literal is assigned in the composer again, so the two gates can "
        f"drift: {reassigned}"
    )


@pytest.mark.parametrize(
    ("value", "usable"),
    [
        (4.4495549999999977e21, False),  # the measured QPLIB_2967 report
        (1e20, False),  # the raw LP sentinel
        (-1e20, False),  # ... and its negation: the test is on magnitude
        (1e19, False),  # the threshold itself is refused
        (9.999e18, True),  # just inside
        (1.0, True),
        (0.0, True),
        (-1e5, True),
        (None, False),
        (float("inf"), False),
        (float("nan"), False),  # every strict test against NaN is False
    ],
)
def test_bound_is_usable_classifies_the_boundary(value, usable):
    assert sv._bound_is_usable(value) is usable


# --------------------------------------------------------------------------
# The class, end to end.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bound", BOXES)
@pytest.mark.parametrize("maximize", [True, False], ids=["max", "min"])
def test_no_sentinel_derived_bound_is_ever_reported(bound, maximize):
    """A forward GUARD, not the reproducer -- measured, these arms pass pre-fix too.

    Stated plainly so the next reader does not mistake a passing sweep for evidence
    that the class reproduces synthetically: a box reaching the sentinel is NOT
    sufficient to trigger #1401. The observed trigger needs the tainted-frontier
    arithmetic a real instance produces, and the discriminating instrument is the
    ``QPLIB_2967`` row of ``scripts/audit_1401_sentinel_bound_reported.py``
    (baseline exit 1, fix exit 0). What discriminates *here* is the predicate block
    above: 13 of this file's tests fail against pre-fix code.

    This sweep earns its place as the assertion that no *future* path re-opens the
    hole on an ordinary large box, which is the shape the class takes.
    """
    res = _sentinel_box_model(bound, maximize).solve(time_limit=10.0)
    b = res.bound
    if b is None:
        return  # "no bound" is the honest report
    b = float(b)
    assert not (np.isfinite(b) and abs(b) >= EFF_INF), (
        f"box=[0,{bound:g}] {'max' if maximize else 'min'}: reported bound={b!r}, "
        f"which is finite but {abs(b) / EFF_INF:.1f}x past the {EFF_INF:g} "
        "effective-infinity refusal -- this is the LP sentinel propagated through "
        f"arithmetic, not a bound anyone proved (gap_certified="
        f"{getattr(res, 'gap_certified', None)})"
    )


def test_the_sweep_actually_reports_bounds():
    """CLAUDE.md §6: prove the sweep above is not vacuous.

    Every row returning ``bound is None`` would make
    ``test_no_sentinel_derived_bound_is_ever_reported`` pass while asserting
    nothing about a *reported* number. At least one box must produce a real,
    finite, sub-threshold bound -- i.e. the fix refuses sentinels without
    degrading into refusing everything.
    """
    reported = 0
    checked = 0
    for bound in BOXES:
        for maximize in (True, False):
            res = _sentinel_box_model(bound, maximize).solve(time_limit=10.0)
            checked += 1
            if res.bound is not None and abs(float(res.bound)) < EFF_INF:
                reported += 1
    assert checked == 2 * len(BOXES), f"checked {checked} rows, expected {2 * len(BOXES)}"
    assert reported > 0, (
        f"all {checked} rows reported no usable bound at all -- the sweep cannot "
        "distinguish 'sentinels are refused' from 'nothing is ever reported'"
    )
