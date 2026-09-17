"""A wall budget must mean the same work on every box (#1204).

The cert panel's budgets were chosen on the machine that generated
``cert-baseline.jsonl``, where the slowest rows certify at roughly half of them
(``tanksize`` 31.0/60 s, ``tls2`` 30.4/60, ``nvs05`` 28.3/60, ``clay0303hfsg``
28.3/60, ``nvs17`` 16.5/30). On a runner ~2x slower they all tip, and whether an
instance certifies becomes a fact about the runner — which then decides which
graduation-gate arms fail, because a row the control lost cannot fail anything
while the same row armed hard-fails whichever arm tips first.

Measured end to end on a box 3.17x the reference (median over 16 equal-node
unrouted probe rows, sd 0.61, load 0.07 — see
``docs/dev/data/README-1204-calibration.md``), the calibrated budget is what puts
the cliff rows back inside the panel:

    tanksize      60 s -> time_limit (13548 nodes)  |  190 s -> optimal  71.4 s, 17139 nodes
    nvs05         60 s -> feasible, obj 6.9358      |  190 s -> optimal 106.6 s, obj 5.4709341

``nvs05``'s nominal incumbent sits 27 % above the true optimum (5.4709) — the
expected shape of an open gap, and exactly the row #1195 had to stop calling a
false certificate. Calibrated, it certifies the true optimum instead.

These tests are the unit-level contract for that mechanism, including its refusals:
an unmeasurable calibration must leave the budgets alone and SAY so, never look
like it worked (CLAUDE.md §6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BENCH_ROOT.parent
for _p in (str(_BENCH_ROOT), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.host_calibration import (  # noqa: E402
    MAX_BUDGET_SCALE,
    MAX_PANEL_WALL_S,
    MIN_CALIBRATION_SAMPLES,
    PROBE_MAX_WALL_S,
    calibration_probe_instances,
    fit_scale_to_panel,
    host_speed_ratio,
    measure_host_scale,
    predicted_panel_wall,
    scale_budgets,
)

pytestmark = [pytest.mark.unit, pytest.mark.correctness]

_CERT_BASELINE = _REPO_ROOT / "docs" / "dev" / "data" / "cert-baseline.jsonl"


def _ref(nodes: int = 10, wall: float = 1.0, status: str = "optimal", route=None) -> dict:
    return {"status": status, "node_count": nodes, "wall_time": wall, "algorithm_route": route}


def _panel(n: int, *, wall: float = 1.0) -> dict[str, dict]:
    return {f"i{k}": _ref(nodes=k + 1, wall=wall) for k in range(n)}


def _same_box(baseline: dict[str, dict], factor: float) -> dict[str, dict]:
    """The same panel re-solved on a box ``factor`` times slower — same tree."""
    return {inst: dict(row, wall_time=row["wall_time"] * factor) for inst, row in baseline.items()}


# --------------------------------------------------------------------------- #
# probe selection
# --------------------------------------------------------------------------- #
def test_probe_rows_are_settled_unrouted_and_inside_the_band():
    baseline = {
        "fast": _ref(wall=0.01),  # below the noise floor
        "slow": _ref(wall=PROBE_MAX_WALL_S + 1),  # too expensive to probe with
        "routed": _ref(wall=1.0, route="convex-route: abstained"),
        "unsettled": _ref(wall=1.0, status="feasible"),
        "good": _ref(wall=1.0),
    }
    assert calibration_probe_instances(baseline) == ["good"]


def test_probe_rows_are_ordered_by_signal_and_capped():
    baseline = {"a": _ref(wall=0.5), "b": _ref(wall=2.0), "c": _ref(wall=1.0)}
    assert calibration_probe_instances(baseline) == ["b", "c", "a"]
    assert calibration_probe_instances(baseline, max_rows=2) == ["b", "c"]


def test_probe_selection_can_be_restricted_to_what_is_vendored():
    baseline = {"a": _ref(wall=2.0), "b": _ref(wall=1.0)}
    assert calibration_probe_instances(baseline, available={"b"}) == ["b"]


def test_the_real_reference_yields_a_usable_probe_that_excludes_the_cliff_rows():
    """Against the committed reference, not a fixture.

    A selection rule that works only on synthetic rows is the #727 lesson (a
    mechanism validated on a proxy that was a no-op on the real class), so this
    reads ``cert-baseline.jsonl`` itself. The five cliff rows must NOT be probe rows:
    the probe's whole point is to be cheap, and those are the expensive ones.
    """
    baseline = {}
    for line in _CERT_BASELINE.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            baseline[row["instance"]] = row
    probe = calibration_probe_instances(baseline)
    assert len(probe) >= MIN_CALIBRATION_SAMPLES, (
        f"the committed reference only offers {len(probe)} probe row(s) — the "
        "calibration could never fire on it"
    )
    for cliff in ("tanksize", "tls2", "nvs05", "clay0303hfsg", "nvs17"):
        assert cliff not in probe, f"{cliff} is a cliff row, not a cheap probe row"
    cost = sum(baseline[i]["wall_time"] for i in probe)
    assert cost < 20.0, f"probe would cost {cost:.1f} reference-seconds — too expensive"


# --------------------------------------------------------------------------- #
# the measurement
# --------------------------------------------------------------------------- #
def test_a_slower_box_scales_the_budgets():
    baseline = _panel(8)
    scale = measure_host_scale(_same_box(baseline, 2.5), baseline, load=(0.1, 0.1, 0.1))
    assert scale.measured
    assert scale.ratio == pytest.approx(2.5)
    assert scale.scale == pytest.approx(2.5)
    assert scale.samples == 8
    assert scale.spread == pytest.approx(0.0)
    assert "x2.50" in scale.reason and "sd 0.00" in scale.reason


def test_a_faster_box_keeps_the_nominal_budgets():
    """Never tighten. A box faster than the reference must not make the panel a
    harder test than the reference ran — that would invent a new cliff."""
    baseline = _panel(8)
    scale = measure_host_scale(_same_box(baseline, 0.4), baseline)
    assert scale.ratio == pytest.approx(0.4)
    assert scale.scale == 1.0
    assert "never tightened" in scale.reason


def test_a_pathologically_slow_box_is_capped_and_says_so():
    baseline = _panel(8)
    scale = measure_host_scale(_same_box(baseline, 50.0), baseline)
    assert scale.scale == MAX_BUDGET_SCALE
    assert "capped" in scale.reason
    assert "not charged as soundness" in scale.reason, (
        "a capped calibration must say what happens to the rows it could not rescue"
    )


def test_too_few_samples_refuses_rather_than_guesses():
    baseline = _panel(MIN_CALIBRATION_SAMPLES - 1)
    scale = measure_host_scale(_same_box(baseline, 3.0), baseline)
    assert not scale.measured
    assert scale.scale == 1.0
    assert "UNAVAILABLE" in scale.reason
    assert str(MIN_CALIBRATION_SAMPLES) in scale.reason


def test_a_probe_that_measured_nothing_cannot_look_calibrated():
    """The failure this whole file is defensive about (CLAUDE.md §6).

    An empty probe — the subprocess died, the instances were not vendored — must
    produce a scale that says so, not a silent 1.0 indistinguishable from "this box
    matches the reference".
    """
    scale = measure_host_scale({}, _panel(8))
    assert scale.scale == 1.0
    assert scale.samples == 0
    assert not scale.measured
    assert "UNAVAILABLE" in scale.reason


def test_a_moved_tree_is_not_a_speed_measurement():
    """Equal node_count is necessary: different trees did different work, so their
    wall ratio is not the box's."""
    baseline = _panel(8)
    moved = {
        i: dict(r, node_count=r["node_count"] + 1, wall_time=r["wall_time"] * 3)
        for i, r in baseline.items()
    }
    assert host_speed_ratio(moved, baseline) == (None, 0)
    assert not measure_host_scale(moved, baseline).measured


def test_a_routed_row_measures_the_router_not_the_box():
    """#1134 Cause 2: a routed row carries a 14-44x inflation that is not hardware."""
    baseline = _panel(8)
    routed = {
        i: dict(r, wall_time=r["wall_time"] * 40, algorithm_route="convex-route")
        for i, r in baseline.items()
    }
    assert host_speed_ratio(routed, baseline) == (None, 0)


def test_the_median_survives_one_wild_row():
    """The probe on the development box returned one row at 0.84 against a median of
    3.17 (``dispatch``, a 0.17 s row — process noise at that scale). A mean would
    have moved; the median must not."""
    baseline = _panel(9)
    rows = _same_box(baseline, 3.0)
    rows["i0"] = dict(rows["i0"], wall_time=baseline["i0"]["wall_time"] * 0.8)
    scale = measure_host_scale(rows, baseline)
    assert scale.ratio == pytest.approx(3.0)
    assert scale.spread > 0.0, "the spread must report the disagreement, not hide it"


# --------------------------------------------------------------------------- #
# applying it
# --------------------------------------------------------------------------- #
def test_every_budget_is_scaled_by_the_same_number():
    budgets = {"a": 60.0, "b": 30.0, "c": 120.0}
    out = scale_budgets(budgets, 2.5)
    assert out == {"a": 150.0, "b": 75.0, "c": 300.0}
    # Per-instance correction would be fitting the budget to the instance — the
    # hand-gating CLAUDE.md §2 rules out, and what _KNOWN_PERF_GATED already is.
    assert {v / budgets[k] for k, v in out.items()} == {2.5}


def test_an_unscaled_call_copies_rather_than_aliases():
    budgets = {"a": 60.0}
    out = scale_budgets(budgets, 1.0)
    assert out == budgets and out is not budgets
    out["a"] = 1.0
    assert budgets["a"] == 60.0


def test_the_gate_measures_once_for_every_panel():
    """One scale for the control and all seven arms.

    Re-measuring per arm would insert a fresh noisy multiplier between an arm and
    its reference — a new version of the asymmetry #1204 is about. Asserted on the
    gate's source because it is a wiring property, not a library one.
    """
    text = (_BENCH_ROOT / "scripts" / "graduation_gate.py").read_text()
    assert text.count("= run_host_calibration()") == 1, (
        "the calibration must be CALLED exactly once per gate run (the def does not count)"
    )
    assert text.count("def run_host_calibration(") == 1
    assert "budget_scale=budget_scale" in text, "the arms do not receive the control's scale"
    assert "GRADGATE_BUDGET_SCALE" in text, "the scale never reaches the panel subprocess"


# --------------------------------------------------------------------------- #
# bounding what the calibration may cost
# --------------------------------------------------------------------------- #
def test_the_predicted_panel_wall_is_work_bounded_not_budget_bounded():
    """A row costs what it needs, unless that is more than its budget."""
    baseline = {"quick": _ref(wall=10.0), "slow": _ref(wall=50.0)}
    budgets = {"quick": 60.0, "slow": 60.0}
    # x2 box: quick needs 20 s, slow needs 100 s but is capped by its 120 s budget.
    assert predicted_panel_wall(baseline, budgets, 2.0, 2.0) == pytest.approx(120.0)
    # x1: each row costs the work it needs, both inside their budgets.
    assert predicted_panel_wall(baseline, budgets, 1.0, 1.0) == pytest.approx(60.0)


def test_an_uncertified_reference_row_is_assumed_to_burn_its_budget():
    """The pessimistic reading is the right one for a cost guard: a row the
    reference itself did not finish has no work estimate to scale."""
    baseline = {"open": _ref(wall=10.0, status="feasible")}
    assert predicted_panel_wall(baseline, {"open": 60.0}, 2.0, 2.0) == pytest.approx(120.0)


def test_an_affordable_scale_is_left_alone():
    baseline = _panel(10, wall=1.0)
    budgets = dict.fromkeys(baseline, 60.0)
    scale = measure_host_scale(_same_box(baseline, 3.0), baseline)
    fitted, note = fit_scale_to_panel(baseline, budgets, scale)
    assert fitted == pytest.approx(3.0)
    assert note is None


def test_an_unaffordable_scale_is_shrunk_and_says_what_it_gave_up():
    """The tail this guard exists for: 8 panels under an unbounded multiplier turn a
    45-minute job into one that times out and returns NO verdict — strictly worse
    than the flaky verdict #1204 is fixing."""
    # 20 rows that each need more than their budget, so every row burns it.
    baseline = {f"i{k}": _ref(nodes=k, wall=90.0) for k in range(20)}
    budgets = dict.fromkeys(baseline, 60.0)
    scale = measure_host_scale(_same_box(baseline, 4.0), baseline)
    fitted, note = fit_scale_to_panel(baseline, budgets, scale, max_panel_wall_s=1500.0)
    assert 1.0 < fitted < scale.scale
    assert predicted_panel_wall(baseline, budgets, fitted, scale.ratio) <= 1500.0 + 1e-6
    assert note is not None and "wall_regression" in note, (
        "a reduced scale must say what happens to the rows it no longer rescues"
    )


def test_the_shrink_never_goes_below_the_nominal_budgets():
    baseline = {f"i{k}": _ref(nodes=k, wall=90.0) for k in range(20)}
    budgets = dict.fromkeys(baseline, 60.0)
    scale = measure_host_scale(_same_box(baseline, 4.0), baseline)
    fitted, _ = fit_scale_to_panel(baseline, budgets, scale, max_panel_wall_s=1.0)
    assert fitted == 1.0, "nominal is the floor — below it the panel is not the panel"


def test_an_unmeasured_calibration_is_not_shrunk():
    baseline = _panel(2)
    scale = measure_host_scale({}, baseline)
    assert fit_scale_to_panel(baseline, dict.fromkeys(baseline, 60.0), scale) == (1.0, None)


def test_the_real_panel_fits_its_ceiling_at_the_maximum_scale():
    """On the committed reference, the cost guard must not bind before the scale cap
    does — otherwise the cap is decorative and every slow box silently runs a
    reduced calibration."""
    baseline = {}
    for line in _CERT_BASELINE.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            baseline[row["instance"]] = row
    budgets = dict.fromkeys(baseline, 60.0)
    predicted = predicted_panel_wall(baseline, budgets, MAX_BUDGET_SCALE, MAX_BUDGET_SCALE)
    assert predicted <= MAX_PANEL_WALL_S, (
        f"a x{MAX_BUDGET_SCALE:g} box predicts {predicted:.0f} s of panel wall against a "
        f"{MAX_PANEL_WALL_S:.0f} s ceiling — one of the two constants is wrong"
    )


def test_budgets_for_instances_outside_the_panel_are_not_charged():
    """The cost guard must price the panel, not the budget map.

    ``_instance_budgets()`` covers global50 plus the perf panel — a strict superset
    of the cert baseline the panel actually solves. Charging every extra instance a
    full budget predicted ~9000 s against the 900 s ceiling in an end-to-end run and
    shrank a measured x3.12 calibration back to x1.0: the guard silently cancelling
    the calibration it was meant to bound. Found by running the real
    ``check_cert_neutrality.main()``, not by a unit test, which is why this one
    exists.
    """
    baseline = {"in_panel": _ref(wall=10.0)}
    budgets = {"in_panel": 60.0, "elsewhere": 60.0, "also_elsewhere": 600.0}
    # Only in_panel counts: 10 s of work at x2 = 20 s, inside its 120 s budget.
    assert predicted_panel_wall(baseline, budgets, 2.0, 2.0) == pytest.approx(20.0)

    scale = measure_host_scale(_same_box(_panel(8), 3.0), _panel(8))
    fitted, note = fit_scale_to_panel(baseline, budgets, scale, max_panel_wall_s=900.0)
    assert fitted == pytest.approx(scale.scale), "an affordable panel must not be shrunk"
    assert note is None
