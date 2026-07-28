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
