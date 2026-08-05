"""#930: the root LP probe's bound must not be thrown away, and must never be
replaced by a worse one.

On the spatial path ``solve_model`` solves the root McCormick LP once as a
keep/discard "probe" for the relaxer, then consumed the answer *only* as a
boolean. The rigorous lower bound it had just proved — inside the time budget,
already paid for — was discarded. When the search then hit its limit with a
bound-less tree, the #138 root-relaxation fallback ran on a starved grant and
whatever weaker value it managed became the reported dual bound.

Measured on ``hda`` at ``time_limit=10`` before the fix::

    8.31s  solve_at_node   lb=-64473.442402437024  status=optimal   (the probe)
   10.42s  SOLVE RETURNED  bound=-141697.43348991545

So this was never merely wasted work: a bound the solver had *proved* was
replaced by one 2.2x looser. Both halves of the fix are exercised here.

The soundness gate is the whole risk surface. The probe box is the FBBT/OBBT
**tightened** root box under ``DISCOPT_ROOT_LP_PROBE_TIGHT`` (default ON), and a
bound proved over a strict subset of the root box need not bound the global
optimum at all — publishing one would be a false certificate, the exact failure
CLAUDE.md §1 makes non-negotiable. Hence exact box equality, and hence the
``_admissible_probe_bound`` tests below, which are the ones that matter.
"""

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver import _admissible_probe_bound, _root_relaxation_lower_bound

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def test_admissible_probe_bound_soundness_gate():
    """A probe bound is admissible only when its box is *exactly* the root box.

    Not "contained in": measured over 10 in-repo instances, of the 8 that
    produced both boxes, 3 (``4stufen``, ``beuster``, ``bchoco06``) had boxes
    that were **not even comparable** — neither a subset of the other — so a
    containment test would buy nothing here while opening a real unsoundness.
    """
    lb = np.array([0.0, -1.0, 2.0])
    ub = np.array([1.0, 3.0, 5.0])
    checks = 0

    # Exact equality on a distinct-but-equal array pair: admissible.
    assert _admissible_probe_bound((-7.5, lb.copy(), ub.copy()), lb, ub) == -7.5
    checks += 1

    # No probe ran.
    assert _admissible_probe_bound(None, lb, ub) is None
    checks += 1

    # Non-finite values are not bounds.
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert _admissible_probe_bound((bad, lb.copy(), ub.copy()), lb, ub) is None
        checks += 1

    # A strict SUBSET of the root box: its optimum need not bound the global
    # optimum. This is the false-certificate case; it must be declined.
    tight_ub = ub.copy()
    tight_ub[1] = 2.0
    assert _admissible_probe_bound((1e9, lb.copy(), tight_ub), lb, ub) is None
    checks += 1

    # A strict SUPERSET is a valid global bound mathematically, but is still
    # declined: the gate is equality, deliberately, so no future caller can
    # smuggle in a box whose relationship to the root was never established.
    wide_lb = lb.copy()
    wide_lb[0] = -10.0
    assert _admissible_probe_bound((1.0, wide_lb, ub.copy()), lb, ub) is None
    checks += 1

    # Incomparable boxes (the 4stufen/beuster/bchoco06 shape).
    other_lb, other_ub = lb.copy(), ub.copy()
    other_lb[0] = -5.0
    other_ub[2] = 4.0
    assert _admissible_probe_bound((1.0, other_lb, other_ub), lb, ub) is None
    checks += 1

    # Dimension mismatch must not raise, and must not be admitted.
    assert _admissible_probe_bound((1.0, lb[:2].copy(), ub[:2].copy()), lb, ub) is None
    checks += 1

    assert checks == 9, f"PROBE NEVER FIRED: only {checks} gate checks executed"


def test_fallback_merges_the_probe_bound_by_max_and_only_on_box_equality():
    """The merge itself, deterministically — no timing, no starved grant.

    ``hda`` is the motivating case but its pre-fix failure was bimodal (2/5
    reps), so it cannot be the primary regression assertion. This drives
    ``_root_relaxation_lower_bound`` directly on a tiny nonconvex model and
    injects a probe value relative to the bound the fallback proves on its own,
    which pins all three arms exactly:

      * a WEAKER probe bound changes nothing (``max`` keeps the fallback's own),
        so re-admitting it can never loosen a reported bound;
      * a TIGHTER one is adopted, which is the whole point of #930;
      * a tighter one over a DIFFERENT box is refused anyway — soundness beats
        tightness, and this arm is the one that would publish a false
        certificate if the gate ever regressed.
    """
    model = dm.Model()
    x = model.continuous("x", lb=-2.0, ub=3.0)
    y = model.continuous("y", lb=-2.0, ub=3.0)
    model.subject_to(x * y <= 1.0)
    model.minimize(x * x + y)

    lb = np.array([-2.0, -2.0])
    ub = np.array([3.0, 3.0])

    base = _root_relaxation_lower_bound(model, lb, ub, 5.0)
    assert base is not None and np.isfinite(base), (
        f"PROBE NEVER FIRED: the fallback proved no bound on its own ({base!r}), "
        "so the three arms below would all be vacuous"
    )

    weaker = _root_relaxation_lower_bound(
        model, lb, ub, 5.0, probe=(base - 100.0, lb.copy(), ub.copy())
    )
    assert weaker == pytest.approx(base), f"a weaker probe bound loosened the result: {weaker!r}"

    tighter = _root_relaxation_lower_bound(
        model, lb, ub, 5.0, probe=(base + 100.0, lb.copy(), ub.copy())
    )
    assert tighter == pytest.approx(base + 100.0), (
        f"the probe's tighter bound was discarded (got {tighter!r}) — this is #930"
    )

    # Same tighter value, box shifted off the root box: must be refused.
    off_box = _root_relaxation_lower_bound(
        model, lb, ub, 5.0, probe=(base + 100.0, lb.copy(), ub.copy() - 1.0)
    )
    assert off_box == pytest.approx(base), (
        f"a bound proved over a DIFFERENT box was admitted (got {off_box!r}); "
        "that is a false certificate, not a tighter bound"
    )


# ``hda`` at an 8 s limit: the discriminating end-to-end case, and NOT the 10 s
# one this issue was opened on. At 10 s the fallback usually has room to re-derive
# the probe's bound itself, so the reported value there depends on machine load —
# measured -141697 in 2/5 reps under one load and -64473 in 3/3 under another. A
# test pinned to 10 s would pass on the pre-fix tree most of the time and so prove
# nothing. At 8 s the budget is gone by the time the probe finishes, the pre-fix
# fallback is skipped entirely for want of remaining time, and the defect is
# unconditional (interleaved A/B, marker-asserted on both trees):
#
#     limit  pre-fix bound          post-fix bound         pre / post wall
#     6 s    None, -141697          -141697, -141697       10.61,6.74 / 6.49,6.11
#     8 s    None, None             -64473.44, -64473.44   11.08,11.42 / 10.48,10.10
#
# So at 8 s the solver had PROVED -64473.44 inside the budget and then reported no
# dual bound whatsoever — the #138 goal ("never report bound=None") violated by the
# very path that exists to uphold it. Recovering it costs no wall time; it is
# already paid for.
_TL_S = 8.0


@pytest.mark.slow
def test_bound_survives_when_the_budget_ends_before_the_fallback():
    """A bound proved inside the budget must be reported, not dropped.

    Pre-fix this returned ``None`` (2/2 reps): ``_rr_remaining <= 0`` skipped the
    root-relaxation fallback, and the probe's bound had nowhere else to go.
    """
    model = from_nl(str(_NL_DIR / "hda.nl"))
    result = model.solve(time_limit=_TL_S)

    assert result.bound is not None, (
        "no dual bound reported, yet the root LP probe proved -64473.44 inside "
        "the budget — the bound was computed and then discarded (#930)"
    )
    assert result.bound >= -100000.0, (
        f"reported dual bound {result.bound!r} is worse than the -64473.44 the root "
        "LP probe proved inside the budget"
    )


@pytest.mark.slow
def test_probe_seeding_flag_is_bound_preserving_here(monkeypatch):
    """``DISCOPT_ROOT_PROBE_SEEDS_FALLBACK`` must not cost the bound on ``hda``.

    The flag lets the fallback's existing rule-2 checkpoint see that a bound is
    already in hand, so it stops re-deriving it — measured 2.72 s of duplicate
    ``solve_at_node`` returning the probe's value to all 17 digits. It is
    default-off and §5 bound-changing (a starved probe's bound can be looser
    than a full separated solve would prove), so this asserts only what the
    motivating instance shows: switching it on does not degrade the bound.
    """
    monkeypatch.setenv("DISCOPT_ROOT_PROBE_SEEDS_FALLBACK", "1")
    model = from_nl(str(_NL_DIR / "hda.nl"))
    result = model.solve(time_limit=_TL_S)

    assert result.bound is not None, "flag ON lost the dual bound outright"
    assert result.bound >= -100000.0, (
        f"flag ON degraded the dual bound to {result.bound!r}; expected the probe-proved -64473.44"
    )
