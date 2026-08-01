"""#902 — the #764 graduation panel must be able to SEE an incumbent-quality regression.

The native spatial kernel graduated default-ON on 2026-07-27 on a panel that could
not detect the regression it shipped. Two independent blind spots:

1. **Instance coverage.** The panel enumerated only
   ``python/tests/data/minlplib_nl`` (66 instances). The other in-repo corpus,
   ``python/tests/data/minlplib`` (81), is *not* a superset — they share 28 — and
   it is the one holding ``nvs17``/``nvs19``/``nvs24``, precisely the family where
   the kernel engages and regresses. Conversely ``tanksize``, the single instance
   whose win carried the net-positive bar, exists ONLY in ``minlplib_nl``. Neither
   directory alone can both justify and falsify the flag.

2. **Metric coverage.** Every cert check required at least one side to be
   ``optimal``: objective agreement needs BOTH optimal, the optimality-regression
   check needs OFF optimal, the feasibility check needs ON optimal. When neither
   run certifies — the common case on hard instances — *nothing fired*. nvs19
   returned ON ``time_limit`` at -315.0 (71% from the reference -1098.4) against
   OFF ``feasible`` at -1097.6 (0.1% off, in 9 nodes), and the panel reported a
   clean pass.

The dual bounds stayed valid throughout, so this was never a soundness failure —
which is why the new gate is `quality_clean`, separate from `cert_clean`. It
blocks graduation regardless, because "the flag is sound" was never the bar
(CLAUDE.md §5 requires net-positive too).

These tests pin both fixes, and — critically — pin that the OLD gates still pass
on the regression, because that is the property that made the miss possible.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PANEL = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "issue764_native_kernel_graduation_panel.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_panel902", _PANEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair(off: dict, on: dict) -> dict:
    return {"off": off, "on": on}


def _nvs19_regression() -> dict:
    """The exact measured shape: neither side optimal, ON far worse."""
    return _pair(
        {
            "status": "feasible",
            "objective": -1097.6,
            "bound": -1104.24,
            "node_count": 9,
            "wall": 20.0,
            "sense": "min",
            "engaged": False,
        },
        {
            "status": "time_limit",
            "objective": -315.0,
            "bound": -5592.5,
            "node_count": 9303,
            "wall": 20.0,
            "sense": "min",
            "engaged": True,
        },
    )


def _tanksize_win() -> dict:
    return _pair(
        {
            "status": "time_limit",
            "objective": None,
            "bound": 1.0,
            "node_count": 5,
            "wall": 60.0,
            "sense": "min",
            "engaged": False,
        },
        {
            "status": "optimal",
            "objective": 2.0,
            "bound": 2.0,
            "node_count": 7,
            "wall": 5.0,
            "sense": "min",
            "engaged": True,
            "incumbent_feasible": True,
            "verified_obj": 2.0,
        },
    )


def test_corpus_is_the_union_of_both_directories():
    m = _load()
    names = m._corpus_instances()
    # Both families must be present, or the panel cannot both justify and falsify.
    for needed in ("nvs17", "nvs19", "nvs24"):
        assert needed in names, f"{needed} missing — the regressing family is unpanelled"
    assert "tanksize" in names, "tanksize missing — the instance carrying net-positive"
    assert len(names) > 100, f"expected the ~119-instance union, got {len(names)}"


def test_instances_resolve_across_both_corpus_dirs():
    m = _load()
    # One from each directory: nvs19 lives in minlplib, tanksize in minlplib_nl.
    assert m._instance_path("nvs19").exists()
    assert m._instance_path("tanksize").exists()
    with pytest.raises(FileNotFoundError):
        m._instance_path("definitely_not_an_instance_902")


def test_quality_gate_catches_the_regression_the_old_gates_missed():
    m = _load()
    rows = {"nvs19": _nvs19_regression(), "tanksize": _tanksize_win()}
    v = m._evaluate(rows, {"nvs19": -1098.4, "tanksize": 2.0})

    # THE key property: the old certification gates still PASS. That is exactly why
    # the regression shipped, and it must stay visible in this test — if a future
    # change makes cert_clean fail here, the separation of concerns has been lost.
    assert v["cert_clean"] is True, "cert gates are about soundness; this is not unsound"

    assert v["quality_clean"] is False, "the new gate must catch a 71%-worse incumbent"
    assert any("nvs19" in s for s in v["quality_violations"])
    assert v["graduate"] is False, "graduation must be blocked by the quality gate alone"


def test_quality_gate_does_not_fire_on_a_genuine_improvement():
    """Precision matters as much as sensitivity. When ON finds a primal that OFF
    missed entirely (the measured nvs24 shape), that is an IMPROVEMENT and must not
    be flagged — a blunt gate that fires on every difference would be ignored."""
    m = _load()
    rows = {
        "nvs24": _pair(
            {
                "status": "time_limit",
                "objective": None,
                "bound": -106418.0,
                "node_count": 2806,
                "wall": 30.0,
                "sense": "min",
                "engaged": False,
            },
            {
                "status": "time_limit",
                "objective": -292.6,
                "bound": -155345.0,
                "node_count": 3019,
                "wall": 30.0,
                "sense": "min",
                "engaged": True,
            },
        ),
        "tanksize": _tanksize_win(),
    }
    v = m._evaluate(rows, {"nvs24": -1033.2, "tanksize": 2.0})
    assert v["quality_clean"] is True, f"false positive: {v['quality_violations']}"


def test_losing_a_primal_is_a_quality_violation():
    m = _load()
    rows = {
        "x": _pair(
            {
                "status": "feasible",
                "objective": 5.0,
                "bound": 1.0,
                "node_count": 3,
                "wall": 1.0,
                "sense": "min",
                "engaged": False,
            },
            {
                "status": "time_limit",
                "objective": None,
                "bound": 1.0,
                "node_count": 99,
                "wall": 1.0,
                "sense": "min",
                "engaged": True,
            },
        ),
        "tanksize": _tanksize_win(),
    }
    v = m._evaluate(rows, {"tanksize": 2.0})
    assert v["quality_clean"] is False
    assert any("PRIMAL LOST" in s for s in v["quality_violations"])


def test_control_clean_panel_still_graduates():
    """The gate must not block a genuinely good flag, or it is useless."""
    m = _load()
    good = _nvs19_regression()
    good["on"]["objective"] = -1097.6  # ON matches OFF
    rows = {"nvs19": good, "tanksize": _tanksize_win()}
    v = m._evaluate(rows, {"nvs19": -1098.4, "tanksize": 2.0})
    assert v["quality_clean"] is True
    assert v["graduate"] is True, "a clean panel with a real win must still graduate"


def test_maximize_sense_is_respected():
    """For a MAXIMIZE model a SMALLER objective is worse; a sense-blind comparison
    would invert the verdict."""
    m = _load()
    rows = {
        "mx": _pair(
            {
                "status": "feasible",
                "objective": 100.0,
                "bound": 200.0,
                "node_count": 3,
                "wall": 1.0,
                "sense": "max",
                "engaged": False,
            },
            {
                "status": "feasible",
                "objective": 10.0,
                "bound": 200.0,
                "node_count": 3,
                "wall": 1.0,
                "sense": "max",
                "engaged": True,
            },
        ),
        "tanksize": _tanksize_win(),
    }
    v = m._evaluate(rows, {"tanksize": 2.0})
    assert v["quality_clean"] is False, "a maximize objective dropping 100 -> 10 is worse"


# ---------------------------------------------------------------------------
# Replication + instability quarantine (#902, second round)
#
# The first fix for load-sensitivity was a blocking load gate ("refuse to start
# until 1-min load < 2.5"). On a real workstation that never runs -- it is a wish,
# not a test. Two alternatives were measured and rejected before landing on
# replication (both recorded in the panel's own comments):
#   * a deterministic max_nodes budget makes solves bit-reproducible but the kernel
#     falls back to Python on a node_limit exit, so the panel compares OFF vs OFF;
#   * a static producer pre-filter drops tanksize, the instance carrying the verdict.
# What survived: re-run the decisive instances, require differences to REPRODUCE,
# and quarantine instances whose replicates disagree.
# ---------------------------------------------------------------------------


def _runs(statuses, objectives, engaged, sense="min"):
    """A list of replicate result dicts for one arm."""
    return [
        {
            "status": s,
            "objective": o,
            "bound": None,
            "node_count": 1,
            "wall": 10.0,
            "sense": sense,
            "engaged": engaged,
        }
        for s, o in zip(statuses, objectives, strict=True)
    ]


def _with_replicates(pair: dict, off_runs: list, on_runs: list, mod) -> dict:
    """Attach a replicate block shaped exactly as the panel's stage 2 writes it."""
    off_stable = mod._statuses_agree(off_runs)
    on_stable = mod._statuses_agree(on_runs)
    pair["replicates"] = {
        "off": off_runs,
        "on": on_runs,
        "off_stable": off_stable,
        "on_stable": on_stable,
        "stable": off_stable and on_stable,
        "off_median_objective": mod._median_objective(off_runs),
        "on_median_objective": mod._median_objective(on_runs),
    }
    return pair


def test_unstable_instance_is_quarantined_not_counted_as_a_regression():
    """An instance whose replicates disagree on status is the MACHINE deciding, not
    the flag. It must be reported as unresolved and must not fire the quality gate --
    otherwise a busy machine manufactures regressions."""
    m = _load()
    # ON flaps between time_limit and optimal across replicates -> unstable.
    pair = _with_replicates(
        _nvs19_regression(),
        _runs(["feasible"] * 3, [-1097.6] * 3, False),
        _runs(["time_limit", "optimal", "time_limit"], [-315.0, -1098.0, -315.0], True),
        m,
    )
    v = m._evaluate({"nvs19": pair}, {"nvs19": -1098.4})
    assert v["quality_clean"] is True, "an unstable instance must not fire the gate"
    assert any("nvs19" in u for u in v["unstable"]), "instability must be REPORTED"
    assert "nvs19" not in v["helped"], "an unstable instance cannot justify the flag either"


def test_reproducible_regression_still_fires_under_replication():
    """The #902 regression is stable, so replication must NOT dilute it -- the whole
    point is to keep real signal while dropping machine noise."""
    m = _load()
    pair = _with_replicates(
        _nvs19_regression(),
        _runs(["feasible"] * 3, [-1097.6] * 3, False),
        _runs(["time_limit"] * 3, [-315.0, -312.0, -315.0], True),
        m,
    )
    v = m._evaluate({"nvs19": pair}, {"nvs19": -1098.4})
    assert v["quality_clean"] is False
    assert v["unstable"] == []
    assert any("reproduced over" in q for q in v["quality_violations"])


def test_a_win_must_hold_in_every_replicate():
    """A flaky win cannot carry net-positive. The bar has historically rested on a
    SINGLE instance (tanksize), so one lucky replicate must not graduate a flag."""
    m = _load()
    flaky = _with_replicates(
        _tanksize_win(),
        _runs(["time_limit"] * 3, [None] * 3, False),
        _runs(["optimal", "time_limit", "optimal"], [2.0, None, 2.0], True),
        m,
    )
    v = m._evaluate({"tanksize": flaky}, {"tanksize": 2.0})
    assert "tanksize" not in v["helped"], "a win that did not reproduce must not count"
    assert v["net_positive"] is False


def test_a_reproducible_win_does_count():
    """Control for the test above: the same win, stable across replicates, still
    graduates. A gate that rejects everything is as useless as one that accepts
    everything."""
    m = _load()
    solid = _with_replicates(
        _tanksize_win(),
        _runs(["time_limit"] * 3, [None] * 3, False),
        _runs(["optimal"] * 3, [2.0] * 3, True),
        m,
    )
    v = m._evaluate({"tanksize": solid}, {"tanksize": 2.0})
    assert "tanksize" in v["helped"]
    assert v["net_positive"] is True and v["graduate"] is True


def test_median_objective_ignores_replicates_with_no_primal():
    m = _load()
    # The None replicate is dropped, leaving two values -> their average.
    assert m._median_objective(_runs(["a", "b", "c"], [None, -5.0, -7.0], True)) == -6.0
    # Odd count after dropping -> the true middle value, not an average.
    assert m._median_objective(_runs(["a", "b", "c", "d"], [None, -5.0, -7.0, -9.0], True)) == -7.0
    # No replicate found a primal at all.
    assert m._median_objective(_runs(["a"], [None], True)) is None
    assert m._median_objective([]) is None


def test_statuses_agree_detects_disagreement():
    m = _load()
    assert m._statuses_agree(_runs(["optimal"] * 3, [1.0] * 3, True)) is True
    assert m._statuses_agree(_runs(["optimal", "time_limit"], [1.0, 1.0], True)) is False
    assert m._statuses_agree([]) is False


def test_blocking_load_gate_is_gone():
    """Regression guard on the DESIGN: the panel must not reacquire a gate that
    blocks on machine quietness. Robustness comes from replication."""
    m = _load()
    assert not hasattr(m, "_await_quiet_machine")
    assert not hasattr(m, "_LOAD_GATE_MAX")
    assert hasattr(m, "_REPLICATES")


# ---------------------------------------------------------------------------
# Spread-vs-effect attribution (#902, third round)
#
# Measured on the 2026-07-31 119-instance run. ``heatexch_gen1`` was the ONLY
# quality violation on the whole corpus and it alone produced GRADUATE: NO. Its ON
# arm returned 167654.27 / 167545.24 / 167654.27 -- a 109.03 spread -- against an
# ON/OFF median difference of exactly 109.03. Status-stable in both arms, so the
# existing quarantine never fired, and the median of that disagreement was reported
# as "reproduced over 3 replicates".
#
# It was not the flag: the kernel never engaged there (binding_called False), the
# producer probe was timed inside a real solve at 0.016 s, and a 3x2 interleaved
# re-measurement outside the panel FLIPPED the sign (OFF worse twice). A median
# without a spread is not a measurement (CLAUDE.md section 9).
# ---------------------------------------------------------------------------


def _heatexch_shape(off_objs, on_objs, m):
    """The measured shape: both arms ``feasible``, kernel declined, minimize."""
    pair = _pair(
        {"status": "feasible", "objective": off_objs[0], "bound": None, "sense": "min"},
        {
            "status": "feasible",
            "objective": on_objs[0],
            "bound": None,
            "sense": "min",
            "engaged": False,
        },
    )
    return _with_replicates(
        pair,
        _runs(["feasible"] * len(off_objs), off_objs, False),
        _runs(["feasible"] * len(on_objs), on_objs, False),
        m,
    )


def test_difference_inside_the_replicate_spread_is_quarantined():
    """The exact measured heatexch_gen1 numbers: effect == spread, so unattributable."""
    m = _load()
    pair = _heatexch_shape(
        [167545.24112846807] * 3,
        [167654.27179214396, 167545.24112846807, 167654.27179214396],
        m,
    )
    v = m._evaluate({"heatexch_gen1": pair}, {})
    assert v["quality_clean"] is True, (
        "a difference the arm cannot reproduce against itself must not fire the gate"
    )
    assert any("heatexch_gen1" in u for u in v["unstable"]), "it must be REPORTED, not dropped"
    assert any("SPREAD" in u for u in v["unstable"]), "the reason must name the spread"


def test_the_same_difference_fires_when_the_arms_reproduce_themselves():
    """Control, and the reason this is a strengthening rather than a loosening: the
    SAME 109-unit regression, with each arm reproducing itself, still fires."""
    m = _load()
    pair = _heatexch_shape(
        [167545.24112846807] * 3,
        [167654.27179214396] * 3,
        m,
    )
    v = m._evaluate({"heatexch_gen1": pair}, {})
    assert v["quality_clean"] is False, "a reproducible regression must still fire"
    assert v["unstable"] == []


def test_a_large_effect_survives_a_small_spread():
    """A 71%-off regression is not quarantined because its arm wobbles by 1%. The
    rule is signal-vs-noise, not exact reproducibility."""
    m = _load()
    pair = _with_replicates(
        _nvs19_regression(),
        _runs(["feasible"] * 3, [-1097.6] * 3, False),
        _runs(["time_limit"] * 3, [-315.0, -312.0, -315.0], True),
        m,
    )
    v = m._evaluate({"nvs19": pair}, {"nvs19": -1098.4})
    assert v["unstable"] == [], "spread 3.0 against effect 782.6 is attributable"
    assert v["quality_clean"] is False


def test_an_unattributable_instance_cannot_be_counted_as_helped_either():
    """Symmetry: quarantine removes an instance from BOTH sides of the verdict."""
    m = _load()
    pair = _pair(
        {"status": "feasible", "objective": 100.0, "bound": None, "sense": "min"},
        {"status": "optimal", "objective": 90.0, "bound": 90.0, "sense": "min", "engaged": True},
    )
    # ON reaches optimal every time, but its objective swings by as much as the
    # claimed improvement, so the improvement is not attributable.
    pair = _with_replicates(
        pair,
        _runs(["feasible"] * 3, [100.0] * 3, False),
        _runs(["optimal"] * 3, [90.0, 100.0, 90.0], True),
        m,
    )
    v = m._evaluate({"x": pair}, {})
    assert "x" not in v["helped"], "an unattributable win must not carry net-positive"
    assert v["net_positive"] is False


def test_gaining_a_primal_is_never_quarantined_as_unattributable():
    """When OFF finds no incumbent at all there is no two-sided objective difference
    to attribute -- nvs24's real shape -- and it must still count."""
    m = _load()
    pair = _pair(
        {"status": "time_limit", "objective": None, "bound": None, "sense": "min"},
        {
            "status": "optimal",
            "objective": -1031.8,
            "bound": -1031.8,
            "sense": "min",
            "engaged": True,
            "incumbent_feasible": True,
            "verified_obj": -1031.8,
        },
    )
    pair = _with_replicates(
        pair,
        _runs(["time_limit"] * 3, [None] * 3, False),
        _runs(["optimal"] * 3, [-1031.8] * 3, True),
        m,
    )
    v = m._evaluate({"nvs24": pair}, {"nvs24": -1033.2})
    assert v["unstable"] == []
    assert "nvs24" in v["helped"]


def test_objective_spread_units():
    m = _load()
    assert m._objective_spread(_runs(["a"] * 3, [1.0, 3.0, 2.0], True)) == 2.0
    # No replicate found a primal: "no primal" is itself reproducible.
    assert m._objective_spread(_runs(["a"] * 3, [None] * 3, True)) == 0.0
    # Some found one and others did not: maximal disagreement, not a small spread.
    assert m._objective_spread(_runs(["a"] * 3, [1.0, None, 1.0], True)) == float("inf")


def test_spread_threshold_is_not_a_tuned_number():
    """The 2026-07-31 verdict does not depend on where in (0, 1] the threshold sits:
    the decisive instance sat at effect/spread = 1.0 and every other decisive instance
    at 0.0 (spread exactly zero in both arms)."""
    m = _load()
    assert 1.0 < m._MIN_EFFECT_TO_SPREAD <= 4.0
