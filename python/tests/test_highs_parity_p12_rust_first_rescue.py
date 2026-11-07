"""highs-parity P1.2: the #1611 linear-node rescue LP runs on the Rust engine first.

``lp_milp_highs.solve_lp_std_rust_first`` solves on the Rust engine and falls to HiGHS
whenever the Rust answer is not decisive. Both answers pass the same certificate ladder.
It is used by ``_linear_node_lp_rescue`` only: P1.2's entry experiment found the two
engines reach different verdicts on 1/94 #1537 arbitration LPs and 24/85 #1869
free-column LPs, so those callers stay HiGHS-first (plan §7).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

H = pytest.importorskip("discopt.solvers.lp_milp_highs")
pytest.importorskip("highspy")

_TESTS = Path(__file__).parent
_FIX = _TESTS / "data" / "highs_parity"


def _cert_instance(name):
    import scipy.sparse as sp

    d = np.load(_TESTS / "data" / "lp_highs_certificate_instances.npz")

    def g(k):
        return d[f"{name}__{k}"]

    a = sp.csc_matrix((g("data"), g("indices"), g("indptr")), shape=tuple(g("shape")))
    return H.StdForm.from_arrays(g("c"), a, g("b"), g("xl"), g("xu"), float(g("obj_const")))


def test_decisive_rust_answer_is_kept():
    out = H.solve_lp_std_rust_first(_cert_instance("e226"), time_limit=60)
    assert out.status == "optimal" and out.gap_certified
    assert out.stats.get("lp/rust_first_decisive") == 1.0
    assert "lp/rust_first_fallback" not in out.stats


def test_undecided_rust_answer_falls_to_highs(monkeypatch):
    from discopt.solvers import lp_std_engine as E

    monkeypatch.setattr(
        E.RustEngine, "solve", lambda self, sf, **k: E.LpRaw("error", "rust:numerical")
    )
    out = H.solve_lp_std_rust_first(_cert_instance("e226"), time_limit=60)
    assert out.status == "optimal" and out.gap_certified
    assert out.stats.get("lp/rust_first_fallback") == 1.0
    assert out.labels["lp/rust_first_fallback"].startswith("rust error")


def _rescue_fixtures():
    man = json.loads((_FIX / "manifest.json").read_text())
    return [f for f in man["fixtures"] if f.get("caller") == "1611_rescue"]


@pytest.mark.parametrize("entry", _rescue_fixtures(), ids=lambda e: e["file"])
def test_rescue_fixture_verdict_unchanged(entry):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cap_p12",
        _TESTS.parents[1] / "discopt_benchmarks" / "scripts" / "capture_highs_parity_fixtures.py",
    )
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    rf = cap.load_fixture(_FIX / entry["file"])["rf"]
    assert np.array_equal(rf["row_lo"], rf["row_hi"])  # captured from solve_lp_std
    sf = H.StdForm.from_arrays(
        rf["c"], rf["A"].tocsc(), rf["row_lo"], rf["col_lo"], rf["col_hi"], rf["obj_const"]
    )
    hi = H.solve_lp_std(sf, time_limit=60, engine="highs")
    rf_out = H.solve_lp_std_rust_first(sf, time_limit=60)
    assert rf_out.status == hi.status
    if hi.bound is not None:
        assert rf_out.bound == pytest.approx(hi.bound, rel=1e-9, abs=1e-9)
