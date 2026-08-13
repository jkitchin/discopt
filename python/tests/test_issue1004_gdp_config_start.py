"""Regression tests for issue #1004 — the GDP constructor's per-candidate start.

#1004 measured, with the integers pinned to a configuration known to be feasible,
that only 12 of 67 starts (B1) and 2 of 6 (#993 C2) produced a feasible point on
``syngas``, and inferred that ``one_hot_config_subnlp`` "rejects ~80% of genuinely
feasible configurations" because it solves one start per candidate.

The inference does not hold, and the reason is structural rather than numerical:
**the constructor does not draw its start from that family.** At a fixed
configuration its start is

    ``zero_start`` = ``clip(0, lb, ub)`` on every continuous slot,
                     the configuration itself on every one-hot and residual
                     binary slot

so on a GDP it is a function of the model and the configuration alone (the one
exception is a *general* integer outside every one-hot row and outside the 0/1
residual set, which keeps its ``x_relax`` value for ``subnlp`` to round). A
per-start detection rate measured over a *family* of starts therefore says nothing
about the constructor's own rate, which
``docs/dev/data/issue1004-gdp-config-start-detection.md`` then measured directly
over the whole ``gdplib_small`` corpus.

These tests pin the two properties that measurement rests on, so that a future
change which reintroduces start-dependence — or spends the budget on k starts per
candidate, which the issue's own budget arithmetic shows is dominated — fails here
rather than silently costing coverage:

1. the per-candidate start is deterministic and independent of the relaxation point;
2. each candidate configuration is tested exactly once (single start per candidate).

Both are pinned on synthetic models gated on detected structure, never on a named
instance (CLAUDE.md §2).
"""

from __future__ import annotations

import discopt._relax.primal_heuristics as ph
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.primal_heuristics import (
    _WAVE_SOLVE_CAP,
    _get_integer_mask,
    _get_variable_bounds,
    _scan_one_hot_rows,
)

#: Continuous box deliberately excludes 0 on both variables, so ``clip(0, lb, ub)``
#: is a *clipped* start (2.0 and -1.0) rather than a vector of zeros. A test that
#: only checked for zeros would pass against a start that ignored the box.
X_LB, X_UB = 2.0, 9.0
Z_LB, Z_UB = -5.0, -1.0


def _offset_box_gdp():
    """Two 2-way disjunctions over a continuous box that does not contain the origin."""
    m = dm.Model("start_probe")
    y = m.binary("y", 4)
    x = m.continuous("x", lb=X_LB, ub=X_UB)
    z = m.continuous("z", lb=Z_LB, ub=Z_UB)
    m.subject_to(y[0] + y[1] == 1, name="d1")
    m.subject_to(y[2] + y[3] == 1, name="d2")
    m.subject_to(x + z >= 1.0 - 10.0 * y[0], name="bigm")
    m.minimize(x - z + y[2])
    return m


def _capture_seeds(model, x_relax, monkeypatch, **kwargs):
    """Run the wave with every sub-NLP stubbed out, returning the seeds it built.

    Returning ``None`` from the stub keeps the wave running to its own bound instead
    of stopping on the first hit; the dive is a separate mechanism and is stubbed to
    empty so nothing downstream contributes seeds.
    """
    seeds: list[np.ndarray] = []

    def _record(_model, seed, **_kw):
        seeds.append(np.asarray(seed, dtype=np.float64).copy())
        return None

    monkeypatch.setattr(ph, "subnlp", _record)
    monkeypatch.setattr(ph, "one_hot_config_dive", lambda *a, **kw: [])
    ph.one_hot_config_subnlp(model, x_relax, **kwargs)
    return seeds


@pytest.mark.smoke
def test_per_candidate_start_is_independent_of_the_relaxation_point(monkeypatch):
    """The premise of #1004's refutation: the start is not drawn from a family.

    Two relaxation points that disagree on every slot — different argmax per group,
    different continuous values — must produce the *same* continuous start, namely
    ``clip(0, lb, ub)``. If this ever fails, a per-start detection rate measured
    over random starts becomes relevant to the constructor again and #1004's
    arithmetic has to be redone.
    """
    m = _offset_box_gdp()
    lb, ub = _get_variable_bounds(m)
    int_mask = _get_integer_mask(m)
    cont = np.nonzero(~int_mask)[0]
    expected = np.clip(0.0, lb, ub)[cont]

    a = np.array([0.9, 0.1, 0.8, 0.2, 7.0, -4.0])
    b = np.array([0.2, 0.8, 0.1, 0.9, 3.0, -2.0])
    seeds_a = _capture_seeds(m, a, monkeypatch)
    seeds_b = _capture_seeds(m, b, monkeypatch)

    # §6: a passing assertion loop that ran zero times is a no-op that reads green.
    assert seeds_a and seeds_b, (
        f"probe vacuous: the wave built {len(seeds_a)} and {len(seeds_b)} seeds; "
        "nothing was compared"
    )
    checked = 0
    for label, seeds in (("A", seeds_a), ("B", seeds_b)):
        for k, seed in enumerate(seeds):
            np.testing.assert_allclose(
                seed[cont],
                expected,
                atol=0.0,
                err_msg=(
                    f"relaxation point {label}, plan {k}: continuous start "
                    f"{seed[cont]} is not clip(0, lb, ub) = {expected}; the "
                    "per-candidate start has become relaxation-point dependent"
                ),
            )
            checked += 1
    assert checked == len(seeds_a) + len(seeds_b)
    assert checked > 1, f"only {checked} start(s) compared; the property is untested"

    # And the continuous half really is distinguishable from the relaxation points
    # that produced it — otherwise the assertion above is trivially satisfied.
    assert not np.allclose(expected, a[cont]) and not np.allclose(expected, b[cont]), (
        "premise stale: clip(0, lb, ub) coincides with a relaxation point, so the "
        "test cannot tell an independent start from a copied one"
    )


@pytest.mark.smoke
def test_each_candidate_configuration_is_tested_exactly_once(monkeypatch):
    """Single start per candidate — the design #1004's budget argument endorses.

    Under a fixed sub-NLP budget ``B``, per-start detection ``p`` and feasible
    fraction ``f``, ``B`` configurations x 1 start expects ``B*f*p`` finds while
    ``B/k`` configurations x ``k`` starts expects ``(B/k)*f*(1-(1-p)^k)``; the ratio
    ``(1-(1-p)^k)/(k*p)`` is ``<= 1`` for every ``k >= 1``. So spending a second
    start on a candidate already tested is strictly dominated by testing one more
    candidate, and this pins that no such second start exists: every sub-NLP the
    wave runs carries a distinct integer fixing.
    """
    m = _offset_box_gdp()
    x_relax = np.array([0.55, 0.45, 0.6, 0.4, 5.0, -3.0])
    seeds = _capture_seeds(m, x_relax, monkeypatch)

    int_mask = _get_integer_mask(m)
    fixings = [tuple(np.round(s[int_mask]).astype(int).tolist()) for s in seeds]
    assert fixings, "probe vacuous: the wave ran no sub-NLP at all"
    assert len(set(fixings)) == len(fixings), (
        f"{len(fixings) - len(set(fixings))} of {len(fixings)} sub-NLP solves repeat "
        "an integer fixing already tested; the wave is spending budget on extra "
        "starts per candidate, which #1004's budget arithmetic shows is dominated"
    )
    # Every fixing is a valid configuration, so a repeat cannot be excused as a
    # different disjunct selection that happens to collide.
    groups = _scan_one_hot_rows(m, int_mask, int(int_mask.size))
    assert groups, "premise stale: no one-hot structure detected on the probe model"
    idx = {j: i for i, j in enumerate(np.nonzero(int_mask)[0].tolist())}
    for fixing in fixings:
        for g in groups:
            assert sum(fixing[idx[j]] for j in g) == 1


@pytest.mark.smoke
def test_the_wave_budget_is_spent_on_candidates_not_on_starts(monkeypatch):
    """The count the budget argument is denominated in stays a *candidate* count.

    ``_WAVE_SOLVE_CAP`` bounds sub-NLP solves. With one start per candidate the two
    coincide, which is what makes the cap a statement about coverage. A k-start
    variant would keep the same solve count while covering ``1/k`` as many
    configurations — the regression this catches.
    """
    m = _offset_box_gdp()
    x_relax = np.array([0.55, 0.45, 0.6, 0.4, 5.0, -3.0])
    seeds = _capture_seeds(m, x_relax, monkeypatch)

    int_mask = _get_integer_mask(m)
    candidates = {tuple(np.round(s[int_mask]).astype(int).tolist()) for s in seeds}
    assert len(seeds) <= _WAVE_SOLVE_CAP, (
        f"{len(seeds)} sub-NLP solves exceeds _WAVE_SOLVE_CAP={_WAVE_SOLVE_CAP}"
    )
    assert len(candidates) == len(seeds), (
        f"{len(seeds)} solves covered only {len(candidates)} configuration(s); the "
        "cap no longer bounds candidate coverage"
    )
