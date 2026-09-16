"""Make a wall-clock budget mean the same amount of *work* on any box (#1204).

The cert panel's budgets are wall-clock seconds, and they were chosen on the
machine that generated ``cert-baseline.jsonl``. There, the slowest rows certify at
roughly half their budget::

    instance        budget   reference wall   fraction
    nvs17             30 s          16.5 s      0.55
    tanksize          60 s          31.0 s      0.52
    tls2              60 s          30.4 s      0.51
    nvs05             60 s          28.3 s      0.47
    clay0303hfsg      60 s          28.3 s      0.47

Any runner about 2x slower tips all five, and then *whether an instance certifies*
is a fact about the runner rather than about the model or the flag under test.
``nvs17`` is already hand-gated in ``_KNOWN_PERF_GATED`` for exactly this reason —
one member of the class papered over individually, which is the signal that the
class wants handling (CLAUDE.md §2).

Downstream that fact decides verdicts. #1204's four graduation-gate runs are the
evidence: two runs of identical PR code failed on *different* four-arm subsets,
and a ``main`` run that drifted MORE than either of them passed, because its
control lost ``tanksize`` to the wall and so disarmed the tripwire. Gate run 29
(2026-09-14) passed the same way — control ``nvs05`` ``feasible``, control
``tanksize`` ``time_limit`` — which also costs the panel 3 of its 52 rows on every
arm, every run: they are reported UNMEASURED and decide nothing.

The fix here is the budget: measure how fast this box is relative to the reference
machine and scale every budget by it. An instance then gets the same *work
allowance* the reference gave it, so its certification is a property of the model
again. It also restores the reference's semantics **below** the top-level stop:
the solver's role-2 sub-budgets are fractions of ``time_limit`` (``_role2_slice``
and friends), so a budget in machine-equivalent seconds makes those slices buy the
same work too — the panel on a slow box becomes more like the reference, not less.

Measured, not assumed (CLAUDE.md §4). On the box this was developed on — 3.17x the
reference's wall over 16 equal-node unrouted rows — the four vendored cliff rows
behave as predicted; the measurement and its kill criterion are recorded in
``docs/dev/data/README-1204-calibration.md``.

What this is NOT: a licence to run without a bound. The scale is clamped to
``[1.0, MAX_BUDGET_SCALE]``, so a faster-than-reference box keeps the nominal
budgets (never make the panel *harder* than the reference) and a pathological
measurement cannot turn a 45-minute job into an overnight one. When the scale
cannot be measured the budgets stay nominal and the run says so — and the residue
is handled by :func:`utils.cert_neutrality.wall_limited_arms`, which refuses to
read a soundness verdict off a row the clock decided.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

#: Minimum number of exactly-reproducing instances before a host-speed ratio is
#: believed. Below this the median is noise, and a speed claim on a handful of
#: sub-second solves is exactly the kind of unfounded timing statement CLAUDE.md §9
#: exists to stop.
MIN_CALIBRATION_SAMPLES = 5

#: Baseline walls at or below this are process noise, not throughput.
CALIBRATION_WALL_FLOOR_S = 0.05

#: Upper bound on the budget multiplier. A box more than this much slower than the
#: reference is not calibrated back into range — its rows stay wall-limited and are
#: reported as such, which is honest, where a 10x budget would silently turn the
#: panel into a different job.
MAX_BUDGET_SCALE = 4.0

#: Probe rows must be fast enough that the probe is cheap and slow enough to carry
#: signal. The band is on the REFERENCE machine's wall, so the probe's own cost is
#: bounded by (sum of these walls) x the very ratio being measured.
PROBE_MIN_WALL_S = 0.05
PROBE_MAX_WALL_S = 3.0
PROBE_MAX_ROWS = 18

#: Wall ceiling for ONE panel, used to bound what a calibration may cost before it
#: runs. A scale is not free: the graduation gate runs 8 panels, and a budget a row
#: fails to certify inside is burned in full, so an unbounded multiplier turns a
#: 45-minute job into an overnight one and the gate returns *no* verdict at all —
#: strictly worse than the flaky verdict this is fixing. The scale is therefore
#: shrunk until the panel's PREDICTED wall fits, and the shrink is reported.
#:
#: 900 s: the committed reference panel is 183 s of certified wall, so this covers a
#: box ~4.9x the reference before it binds at all — past MAX_BUDGET_SCALE.
MAX_PANEL_WALL_S = 900.0


@dataclass(frozen=True)
class HostScale:
    """The budget multiplier for this box, and the evidence behind it.

    ``reason`` is always populated, including when nothing could be measured. A
    calibration that silently did nothing is the failure mode CLAUDE.md §6 is about:
    the caller prints this string, so a run can never *look* calibrated when it was
    not.
    """

    scale: float
    ratio: float | None
    samples: int
    spread: float | None
    load: tuple[float, float, float] | None
    reason: str

    @property
    def measured(self) -> bool:
        return self.ratio is not None


def host_speed_ratio(
    new_rows: dict[str, dict], baseline: dict[str, dict]
) -> tuple[float | None, int]:
    """Median ``wall_new / wall_baseline`` over **unrouted** instances with an
    identical node_count.

    Returns ``(ratio, n_samples)``; ``ratio`` is None when fewer than
    :data:`MIN_CALIBRATION_SAMPLES` instances qualify. Equal node counts are
    *necessary* for this to be a speed measurement rather than a work measurement —
    the two runs explored the same tree, so the wall ratio is the machines'.

    They are not *sufficient*, which #1134's own Cause 2 is the proof of: an
    auto-routed algorithm that runs to a budget checkpoint and then abstains spends
    wall outside the counted tree, so `cvxnonsep_nsig30` (165 → 165 nodes,
    1.12 s → 49.6 s), `fac2` (39 → 39, 2.77 → 38.9) and `cvxnonsep_psig30` (89 → 89,
    0.41 → 8.5) all clear the equal-node filter carrying a 14-44x inflation that is
    not the box. Worse, the route's price is a *fraction of the wall-clock budget*
    (`_CONVEX_ROUTE_BUDGET_FRACTION`), i.e. the same number of seconds on a fast box
    and a slow one, so a row routed on **both** sides compresses the ratio toward 1
    instead of inflating it. Either way the row measures the router, not the
    machine, and is excluded — which is what `algorithm_route` (added by #1134 to
    `SolveResult`) is here for. Reference rows predate the field and read as
    unrouted, which is correct: they were generated before the route was reachable.

    Baseline walls at or below :data:`CALIBRATION_WALL_FLOOR_S` are excluded — at
    that scale the row is process noise, not throughput.

    Lives here rather than in ``check_cert_neutrality`` (its original home, which
    re-exports it) because #1204 made it load-bearing for both gate scripts.
    """
    ratios = _sample_ratios(new_rows, baseline)
    if len(ratios) < MIN_CALIBRATION_SAMPLES:
        return None, len(ratios)
    return statistics.median(ratios), len(ratios)


def _sample_ratios(new_rows: dict[str, dict], baseline: dict[str, dict]) -> list[float]:
    ratios: list[float] = []
    for inst, base in baseline.items():
        new = new_rows.get(inst)
        if new is None or new.get("node_count") != base.get("node_count"):
            continue
        if new.get("algorithm_route") or base.get("algorithm_route"):
            continue
        wb, wn = base.get("wall_time"), new.get("wall_time")
        if wb is None or wn is None or wb <= CALIBRATION_WALL_FLOOR_S:
            continue
        ratios.append(wn / wb)
    return ratios


def calibration_probe_instances(
    baseline: dict[str, dict],
    *,
    available: set[str] | None = None,
    max_rows: int = PROBE_MAX_ROWS,
) -> list[str]:
    """The rows to solve in order to measure this box, cheapest useful set first.

    Chosen from the reference's own numbers, so the choice is deterministic and the
    probe's cost is known before it runs:

    * **settled** on the reference (``optimal``) — a row the reference itself cut
      off at a budget cannot produce an equal-node comparison;
    * **unrouted** — a routed row measures the router (see :func:`host_speed_ratio`);
    * reference wall inside ``[PROBE_MIN_WALL_S, PROBE_MAX_WALL_S]`` — below the
      band a row is process noise, above it the probe stops being cheap.

    Ordered by reference wall **descending**: the longest rows inside the band carry
    the most signal per second spent, so a truncated probe keeps the best samples.
    ``available`` optionally restricts to the instances actually vendored here.
    """
    cand = [
        inst
        for inst, row in baseline.items()
        if row.get("status") == "optimal"
        and not row.get("algorithm_route")
        and row.get("wall_time") is not None
        and PROBE_MIN_WALL_S < float(row["wall_time"]) <= PROBE_MAX_WALL_S
        and (available is None or inst in available)
    ]
    cand.sort(key=lambda inst: (-float(baseline[inst]["wall_time"]), inst))
    return cand[:max_rows]


def measure_host_scale(
    probe_rows: dict[str, dict],
    baseline: dict[str, dict],
    *,
    max_scale: float = MAX_BUDGET_SCALE,
    load: tuple[float, float, float] | None = None,
) -> HostScale:
    """Turn a probe run into the budget multiplier for this box.

    ``probe_rows`` are rows produced **on this box, flag-OFF, at nominal budgets**
    for (a subset of) ``baseline``'s instances. Flag-OFF matters: a flag that moves
    the tree breaks the equal-node filter and would leave the probe with no samples.

    The spread is reported alongside the median because a timing claim without one
    is not a measurement (CLAUDE.md §9); it does not veto the calibration, since the
    median is the robust statistic and refusing on a wide spread would leave the
    cliff exactly where it is.
    """
    if load is None:
        try:
            load = os.getloadavg()
        except (OSError, AttributeError):  # not available on every platform
            load = None
    ratios = _sample_ratios(probe_rows, baseline)
    n = len(ratios)
    if n < MIN_CALIBRATION_SAMPLES:
        return HostScale(
            scale=1.0,
            ratio=None,
            samples=n,
            spread=None,
            load=load,
            reason=(
                f"host-speed calibration UNAVAILABLE: {n} of the {len(probe_rows)} probe "
                f"row(s) reproduced their node_count exactly while unrouted, need "
                f"{MIN_CALIBRATION_SAMPLES} — budgets left at nominal, so wall-limited "
                f"rows stay possible and are reported as such (#1204)"
            ),
        )
    ratio = statistics.median(ratios)
    spread = statistics.stdev(ratios) if n > 1 else 0.0
    scale = min(max_scale, max(1.0, ratio))
    if ratio <= 1.0:
        tail = "at or faster than the reference — budgets stay nominal, never tightened"
    elif scale < ratio:
        tail = (
            f"capped at x{max_scale:g} (MAX_BUDGET_SCALE) — rows may still hit the wall "
            f"and are reported, not charged as soundness"
        )
    else:
        tail = f"budgets scaled x{scale:.2f} so each row gets the reference's work allowance"
    return HostScale(
        scale=scale,
        ratio=ratio,
        samples=n,
        spread=spread,
        load=load,
        reason=(
            f"host-speed calibration: this box is x{ratio:.2f} the reference machine's "
            f"wall (median over {n} equal-node unrouted row(s), sd {spread:.2f}"
            + (f", load {load[0]:.2f}" if load else "")
            + f"); {tail}"
        ),
    )


def predicted_panel_wall(
    baseline: dict[str, dict], budgets: dict[str, float], scale: float, ratio: float
) -> float:
    """What one panel is expected to cost at ``scale``, in seconds on THIS box.

    A row costs the lesser of its budget and the work it actually needs: a row that
    certified on the reference in ``w`` seconds is expected to take ``ratio * w``
    here, and a row that needs more than its budget burns the budget. A reference
    row that did not certify has no ``w`` to scale, so it is assumed to burn its
    whole budget — the pessimistic reading, which is the right one for a guard.

    Checked against the measured cliff rows at ratio 3.172 (predicted → actual):
    ``tanksize`` 98 s → 71 s, ``clay0303hfsg`` 90 s → 56 s, ``nvs05`` 90 s → 107 s.
    Right to about a factor of 1.3 either way on individual rows, which is what a
    cost *guard* needs; it is not a wall-time claim about any one instance.

    The sum runs over ``baseline`` — the rows the panel actually solves — not over
    ``budgets``, which is ``_instance_budgets()``'s global50-plus-perf-panel map and
    a strict superset of it. Iterating the budgets instead charged every instance
    the panel does not run a full unscaled budget, which in an end-to-end run
    predicted ~9000 s against the 900 s ceiling and silently shrank a measured x3.12
    calibration back to x1.0 — a cost guard that quietly cancelled the thing it was
    guarding. Pinned by
    ``test_1204_host_calibration.test_budgets_for_instances_outside_the_panel_are_not_charged``.
    """
    total = 0.0
    for inst, row in baseline.items():
        budget = budgets.get(inst)
        if budget is None:
            continue
        scaled = float(budget) * scale
        wall = row.get("wall_time")
        if wall is None or row.get("status") != "optimal":
            total += scaled
        else:
            total += min(scaled, ratio * float(wall))
    return total


def fit_scale_to_panel(
    baseline: dict[str, dict],
    budgets: dict[str, float],
    scale: HostScale,
    *,
    max_panel_wall_s: float = MAX_PANEL_WALL_S,
) -> tuple[float, str | None]:
    """Shrink ``scale`` until one panel's predicted wall fits ``max_panel_wall_s``.

    Returns ``(scale, note)``; ``note`` is None when nothing was shrunk, and
    otherwise says what was given up. Never shrinks below 1.0: the nominal budgets
    are the floor, and a box slow enough to overrun them at scale 1.0 is a box whose
    rows are going to be wall-limited whatever this returns — which is a reported
    ``wall_regression``, not a soundness fault.
    """
    if not scale.measured or scale.scale <= 1.0:
        return scale.scale, None
    ratio = scale.ratio or scale.scale
    if predicted_panel_wall(baseline, budgets, scale.scale, ratio) <= max_panel_wall_s:
        return scale.scale, None
    lo, hi = 1.0, scale.scale
    for _ in range(40):  # bisection; the prediction is monotone in the scale
        mid = 0.5 * (lo + hi)
        if predicted_panel_wall(baseline, budgets, mid, ratio) <= max_panel_wall_s:
            lo = mid
        else:
            hi = mid
    predicted = predicted_panel_wall(baseline, budgets, scale.scale, ratio)
    return lo, (
        f"budget scale reduced x{scale.scale:.2f} -> x{lo:.2f}: the full scale predicts "
        f"{predicted:.0f} s of panel wall against a {max_panel_wall_s:.0f} s ceiling. "
        f"Rows needing more than the reduced budget stay wall-limited and are reported "
        f"as wall_regression, never as soundness (#1204)"
    )


def scale_budgets(budgets: dict[str, float], scale: float) -> dict[str, float]:
    """Apply a :class:`HostScale` multiplier to every per-instance budget.

    Every instance is scaled by the same number, deliberately. A per-instance
    correction would be fitting the budget to the instance — the hardcoded
    special-casing CLAUDE.md §2 rules out, and the thing ``_KNOWN_PERF_GATED``'s
    single hand-gated row already is.
    """
    if scale == 1.0:
        return dict(budgets)
    return {inst: float(budget) * scale for inst, budget in budgets.items()}
