"""Regression tests for issue #823 — the disjunctive (GDP) primal-constructor gap.

On a big-M reformulated GDP the indicator binaries are partitioned by
``sum_k y_k == 1`` rows, one per disjunction. Two existing constructors both
decline that structure, so the class returns *no incumbent at all*:

* ``enumerate_binary_seeds_subnlp`` enumerates every 0/1 assignment and therefore
  self-gates off above ``max_binaries=4``. Measured on the GDPlib small set, the
  big-M models carry 6..138 binaries, so the root disjunct cover never runs.
* plain ``subnlp`` rounds each binary independently to nearest, which does not
  respect ``sum_k y_k == 1``: indicators reading ``(0.4, 0.35, 0.25)`` round to
  *all zeros* and ``(0.6, 0.55)`` rounds to *two ones*. Either fixing contradicts
  a constraint the model states outright, so the sub-NLP is infeasible.

``one_hot_config_subnlp`` selects one disjunct per row instead (per-group argmax
plus bounded least-confident flips), which is valid by construction.

These tests pin the structure detection and the constructor directly — fast and
deterministic, gated on detected structure rather than on any problem name.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.primal_heuristics import (
    _detect_one_hot_groups,
    _residual_assignments,
    _scan_one_hot_rows,
    one_hot_config_subnlp,
    subnlp,
)


def _uneven_disjunction_model():
    """Two disjunctions of *different* sizes (3-way and 2-way) over one binary vec.

    y[0..2] is one disjunction, y[3..4] another. The continuous x is driven to a
    value determined by which disjunct is active, and the objective prefers the
    third disjunct of the first group.
    """
    m = dm.Model("uneven")
    y = m.binary("y", 5)
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(y[0] + y[1] + y[2] == 1, name="d1")
    m.subject_to(y[3] + y[4] == 1, name="d2")
    # x is pushed up unless the 3rd disjunct of the first group is active.
    m.subject_to(x >= 5.0 - 4.0 * y[2], name="lo")
    m.minimize(x + y[3])
    return m, y, x


def _equal_size_model():
    """Two disjunctions of the SAME size — what the #280 swap detector accepts."""
    m = dm.Model("even")
    y = m.binary("y", 4)
    m.subject_to(y[0] + y[1] == 1, name="d1")
    m.subject_to(y[2] + y[3] == 1, name="d2")
    m.minimize(y[0] + y[2])
    return m


@pytest.mark.smoke
def test_scan_finds_unequal_size_groups_the_swap_detector_rejects():
    """The #823 structural gap: unequal-size disjunctions detect as zero groups.

    ``_detect_one_hot_groups`` requires every group be the same size (the swap
    move pairs members by sorted position). A GDP whose disjunctions have
    differing numbers of disjuncts therefore reads as "no one-hot structure",
    which is correct for the swap move and wrong for disjunct selection.
    """
    m, _, _ = _uneven_disjunction_model()
    mask = np.zeros(6, dtype=bool)
    mask[:5] = True  # y[0..4] binary, x continuous

    scanned = _scan_one_hot_rows(m, mask, mask.size)
    assert [len(g) for g in scanned] == [3, 2], scanned

    # The swap-move detector rejects the very same model.
    assert _detect_one_hot_groups(m, mask, mask.size) == []


@pytest.mark.smoke
def test_equal_size_detection_is_unchanged():
    """The refactor must not alter #280's swap-move detection."""
    m = _equal_size_model()
    mask = np.ones(4, dtype=bool)
    groups = _detect_one_hot_groups(m, mask, mask.size)
    assert len(groups) == 2
    assert all(len(g) == 2 for g in groups)
    assert groups == _scan_one_hot_rows(m, mask, mask.size)


@pytest.mark.smoke
def test_independent_rounding_produces_an_invalid_configuration():
    """Pin the mechanism: nearest-rounding a flat disjunction sets NO indicator.

    This is why the class returns no incumbent — the fixing contradicts
    ``sum_k y_k == 1`` before the NLP is even consulted.
    """
    x_relax = np.array([0.4, 0.35, 0.25, 0.5, 0.5, 3.0])
    rounded = np.round(x_relax[:3])
    assert rounded.sum() == 0.0, "expected an all-zero rounding of the 3-way disjunction"

    # And the constructor's rule picks exactly one, by construction.
    m, _, _ = _uneven_disjunction_model()
    mask = np.zeros(6, dtype=bool)
    mask[:5] = True
    for g in _scan_one_hot_rows(m, mask, mask.size):
        pick = max(g, key=lambda j: x_relax[j])
        assert sum(1 for j in g if j == pick) == 1


@pytest.mark.smoke
def test_config_subnlp_finds_a_point_where_plain_subnlp_cannot():
    """The regression: valid disjunct selection yields an incumbent, rounding does not."""
    m, _, _ = _uneven_disjunction_model()
    # Indicators deliberately fractional so nearest-rounding zeroes both groups.
    x_relax = np.array([0.4, 0.35, 0.25, 0.45, 0.4, 3.0])

    assert subnlp(m, x_relax) is None, "plain rounding should fix an invalid configuration"

    found = one_hot_config_subnlp(m, x_relax)
    assert found, "disjunct selection should produce at least one feasible point"
    for x, obj in found:
        assert np.isfinite(obj)
        # Every returned point satisfies both one-hot rows exactly.
        assert abs(x[0] + x[1] + x[2] - 1.0) < 1e-6
        assert abs(x[3] + x[4] - 1.0) < 1e-6


@pytest.mark.smoke
def test_config_subnlp_is_a_noop_without_one_hot_structure():
    """Generality: gated on detected structure, never on a name."""
    m = dm.Model("plain")
    z = m.binary("z", 3)
    m.subject_to(z[0] + z[1] + z[2] <= 2, name="c")
    m.minimize(z[0] + z[1] + z[2])
    assert one_hot_config_subnlp(m, np.array([0.4, 0.4, 0.4])) == []


@pytest.mark.smoke
def test_config_subnlp_respects_an_expired_deadline():
    """A past deadline must stop it before the first sub-NLP solve."""
    import time

    m, _, _ = _uneven_disjunction_model()
    x_relax = np.array([0.4, 0.35, 0.25, 0.45, 0.4, 3.0])
    assert one_hot_config_subnlp(m, x_relax, deadline=time.perf_counter() - 1.0) == []


@pytest.mark.smoke
def test_residual_assignments_are_ordered_bounded_and_deterministic():
    """Binaries outside every disjunction get their own bounded, ordered search.

    Measured on GDPlib, this is what separates a model the constructor helps from
    one it does not: valid disjunct selection alone suffices on batch_processing
    (one-hot rows cover 138/138 binaries) but not on cstr (15/20) — where
    searching the 5 uncovered binaries turns "no incumbent at all" into a feasible
    3.13020 against a true optimum of 3.06201.
    """
    x_relax = np.array([0.9, 0.2, 0.6])
    got = _residual_assignments([0, 1, 2], x_relax, limit=16)

    # Most-informed first: the relaxation's own nearest rounding.
    assert got[0] == (1.0, 0.0, 1.0)
    # Then the two homogeneous fixings — a big-M indicator's off/on values.
    assert (0.0, 0.0, 0.0) in got[:3] and (1.0, 1.0, 1.0) in got[:3]
    # Exhaustive for a small residual, no duplicates, and bounded by the limit.
    assert len(got) == len(set(got)) == 8
    # Deterministic: a heuristic that varies run to run makes node counts
    # unreproducible, which would break the §5 bound-neutral comparison.
    assert got == _residual_assignments([0, 1, 2], x_relax, limit=16)

    # No residual is still one (empty) assignment, so the cross product is never
    # empty and the group search still runs.
    assert _residual_assignments([], x_relax, limit=16) == [()]

    # A large residual samples instead of enumerating, and still respects limit.
    big = _residual_assignments(list(range(12)), np.zeros(12), limit=10)
    assert len(big) == len(set(big)) == 10


@pytest.mark.smoke
def test_flag_is_default_on_with_an_opt_out():
    """§5: the constructor graduated default-ON, and the opt-out still works.

    It shipped default-OFF pending its differential panel. That panel ran on
    2026-08-12 over the twelve GDPlib models under 500 variables and passed both
    bars (72 checks clean; two models gained an incumbent where 120 s of search
    found none, one gained a certificate) — recorded in
    ``docs/dev/data/issue993-gdp-config-primal-graduation.md``. §5 requires the
    ``=0`` opt-out and the legacy path to survive graduation, so both halves are
    pinned here, not just the new default.
    """
    import os

    from discopt.solver import _gdp_config_primal_enabled

    saved = os.environ.pop("DISCOPT_GDP_CONFIG_PRIMAL", None)
    try:
        assert _gdp_config_primal_enabled() is True, "graduated: unset must mean ON"
        for off in ("0", "off", "false", "no", ""):
            os.environ["DISCOPT_GDP_CONFIG_PRIMAL"] = off
            assert _gdp_config_primal_enabled() is False, f"{off!r} must opt out"
        for on in ("1", "true", "yes", "on"):
            os.environ["DISCOPT_GDP_CONFIG_PRIMAL"] = on
            assert _gdp_config_primal_enabled() is True, f"{on!r} must opt in"
    finally:
        os.environ.pop("DISCOPT_GDP_CONFIG_PRIMAL", None)
        if saved is not None:
            os.environ["DISCOPT_GDP_CONFIG_PRIMAL"] = saved


def test_constructor_gets_a_bounded_share_of_the_budget():
    """A root constructor must be cheap when it fails.

    Handed the whole remaining budget, the search cost batch_processing 71 % of
    its nodes (307 OFF -> 89 ON) while finding nothing on it: the models it
    cannot help paid for the one it can. The deadline it receives is a bounded
    share of what is left, never past the caller's own deadline.
    """
    from discopt.solver import (
        _GDP_CONFIG_BUDGET_CAP_S,
        _GDP_CONFIG_BUDGET_FRACTION,
        _gdp_config_deadline,
    )

    # A share of the remaining budget, not all of it.
    assert _gdp_config_deadline(100.0, 0.0) == pytest.approx(
        min(100.0 * _GDP_CONFIG_BUDGET_FRACTION, _GDP_CONFIG_BUDGET_CAP_S)
    )
    # The absolute cap binds on a long budget.
    assert _gdp_config_deadline(10_000.0, 0.0) == pytest.approx(_GDP_CONFIG_BUDGET_CAP_S)
    # Never past the caller's deadline, however short it is.
    assert _gdp_config_deadline(1.0, 0.0) <= 1.0
    # An already-expired budget stays expired rather than being extended.
    assert _gdp_config_deadline(5.0, 9.0) == 5.0
