"""Gap plan P0: the §5 graduation panel's per-instance bars (``d2_flag_panel.evaluate``).

A summed certification bar let ``DISCOPT_MILP_CUT_LP`` graduate while ``enlight8`` went
3/3 -> 0/3, and the coin-flip row ``p0548`` alone decided three verdicts. These tests pin
the corrected scoring on synthetic rows, and that re-scoring the recorded panels gives the
verdicts recorded in ``docs/dev/highs-parity-d2-gap-plan.md`` (P0 results).
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "discopt_benchmarks" / "scripts" / "d2_flag_panel.py"


def _load():
    spec = importlib.util.spec_from_file_location("d2_flag_panel", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(cert, obj=10.0, bound=10.0, nodes=100):
    return {"status": "optimal" if cert else "time_limit", "gap_certified": cert,
            "objective": obj, "bound": bound if cert else 5.0, "nodes": nodes,
            "max_viol": 0.0, "fired": {}}  # fmt: skip


def _rows(name, off_certs, on_certs, opt=10.0):
    return [{"name": name, "rep": k, "opt": opt, "maximize": False, "problems": {},
             "off": _arm(o), "on": _arm(n)} for k, (o, n) in enumerate(zip(off_certs, on_certs))]  # fmt: skip


@pytest.mark.smoke
def test_an_instance_lost_in_every_rep_fails_despite_a_summed_gain():
    d2 = _load()
    rows = _rows("lost", [1, 1, 1], [0, 0, 0])
    for k in range(4):  # +4 certifications elsewhere: the summed 2a would pass
        rows += _rows(f"gain{k}", [0, 0, 0], [1, 1, 1])
    v = d2.evaluate(rows)
    assert v["bar2"]["2a_solved_on_ge_off"]
    assert v["bar1b_cert_lost_every_rep"] == ["lost"]
    assert not v["graduate"]


@pytest.mark.smoke
def test_bimodal_rows_do_not_decide_2a():
    d2 = _load()
    rows = _rows("steady", [1, 1, 1], [1, 1, 1]) + _rows("coin", [1, 1, 0], [0, 0, 0])
    v = d2.evaluate(rows)
    assert "coin" in v["bimodal_off"]
    assert v["solved"] == {"off": 3, "on": 3}  # the coin row is outside the sums
    assert v["bar2"]["2a_solved_on_ge_off"]
    # a known cross-panel coin flip is set aside even when this panel saw it 3/3
    v2 = d2.evaluate(_rows("p0548", [1, 1, 1], [1, 1, 0]))
    assert "p0548" in v2["bimodal_off"] and v2["solved"] == {"off": 0, "on": 0}


@pytest.mark.parametrize(
    "path, graduate, lost",
    [
        ("highs-parity/d2-analysis/w1_cut_lp_panel_r3.json.gz", False, ["enlight8"]),
        ("highs-parity/d2-analysis/eng_incr_pricing_panel_r3.json.gz", True, []),
        ("highs-parity/d2-analysis/tol_robust_panel_v2_r3.json.gz", False, []),
        ("python-milp-retirement/coef_entry_miplib_panel_r3.json.gz", True, []),
    ],
)
def test_rescoring_the_recorded_panels(path, graduate, lost):
    d2 = _load()
    with gzip.open(ROOT / "docs" / "dev" / "data" / path, "rt") as fh:
        rows = json.loads(fh.read())["rows"]
    v = d2.evaluate(rows)
    assert v["graduate"] is graduate, (path, v["bar2"], v["bar1b_cert_lost_every_rep"])
    assert v["bar1b_cert_lost_every_rep"] == lost
