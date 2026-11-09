"""highs-parity P1.2 -> P6: the #1611 linear-node rescue LP runs on the in-house engine
only.

P1.2 solved it on the Rust engine and fell back to HiGHS whenever the Rust answer was not
decisive. P6 measured that fallback over every test that reaches the rescue
(``docs/dev/data/highs-parity/p6-node-bound-lp/``): 7 of 7 real rescue LPs were decisive
on Rust; the only non-decisive one was this file's own synthetic engine failure. The
fallback was removed (owner: a non-decisive Rust answer is a place to harden, not to
hand to HiGHS). An undecided answer now leaves the node open, which is the rescue's
documented contract ("nothing is inferred from an unverified outcome").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

H = pytest.importorskip("discopt.solvers.lp_milp_highs")

_TESTS = Path(__file__).parent
_FIX = _TESTS / "data" / "highs_parity"


def _cert_instance(name):
    import scipy.sparse as sp

    d = np.load(_TESTS / "data" / "lp_highs_certificate_instances.npz")

    def g(k):
        return d[f"{name}__{k}"]

    a = sp.csc_matrix((g("data"), g("indices"), g("indptr")), shape=tuple(g("shape")))
    return H.StdForm.from_arrays(g("c"), a, g("b"), g("xl"), g("xu"), float(g("obj_const")))


def test_rescue_engine_needs_no_highs(monkeypatch):
    """The rescue's LP seam (``solve_lp_std`` on its default engine) certifies the
    instance with ``highspy`` blocked."""
    import sys

    monkeypatch.setitem(sys.modules, "highspy", None)
    out = H.solve_lp_std(_cert_instance("e226"), time_limit=60)
    assert out.status == "optimal" and out.gap_certified
    assert sys.modules["highspy"] is None


def test_undecided_rust_answer_leaves_the_node_open(monkeypatch):
    """A non-decisive engine answer is no longer handed to HiGHS: the rescue returns
    ``"none"`` (the caller keeps the node's sentinel) and infers nothing."""
    from discopt import solver as S
    from discopt.solvers import lp_std_engine as E

    monkeypatch.setattr(
        E.RustEngine, "solve", lambda self, sf, **k: E.LpRaw("error", "rust:numerical")
    )
    sf = _cert_instance("e226")
    verdict = S._linear_node_lp_rescue(sf, sf.n, None, sf.xl, sf.xu, time_limit=60)
    assert verdict == ("none", None, None)


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
    pytest.importorskip("highspy")
    hi = H.solve_lp_std(sf, time_limit=60, engine="highs")
    rf_out = H.solve_lp_std(sf, time_limit=60)
    assert rf_out.status == hi.status
    if hi.bound is not None:
        assert rf_out.bound == pytest.approx(hi.bound, rel=1e-9, abs=1e-9)
