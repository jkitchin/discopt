"""highs-parity P1.1: ``solve_lp_std`` behind an engine seam, HiGHS default, Rust measured.

The certificate ladder reads only the standard form and the vectors an engine returns, so
the raw solve moved behind ``discopt.solvers.lp_std_engine``. These pin:

* the default engine is HiGHS, ``DISCOPT_LP_STD_ENGINE`` selects, a typo raises;
* the Rust engine reaches the same *verdicts* through the same ladder on the frozen
  certificate instances of the routing work (e226 needs exact dual correction from a
  float dual; klein1 and forest6 are infeasible), never a different answer;
* a Rust ``optimal`` is certified, and carries the Rust route marker rather than the
  HiGHS route's.

The exact-identity check of the HiGHS engine (old code vs new on the L/INF panels'
standard forms: 37/37 identical in every field) is recorded in the plan (P1.1); it needs
the pre-seam module, so it is not a unit test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

H = pytest.importorskip("discopt.solvers.lp_milp_highs")
pytest.importorskip("highspy")

_DATA = Path(__file__).parent / "data" / "lp_highs_certificate_instances.npz"


def _instance(name):
    d = np.load(_DATA)

    def g(k):
        return d[f"{name}__{k}"]

    a = sp.csc_matrix((g("data"), g("indices"), g("indptr")), shape=tuple(g("shape")))
    return H.StdForm.from_arrays(g("c"), a, g("b"), g("xl"), g("xu"), float(g("obj_const")))


def test_default_engine_is_rust_and_env_selects(monkeypatch):
    # highs-parity P6.4 (2026-11-07): the seam defaults to the in-house engine.
    monkeypatch.delenv("DISCOPT_LP_STD_ENGINE", raising=False)
    assert H._lp_engine(None).name == "rust"
    monkeypatch.setenv("DISCOPT_LP_STD_ENGINE", "highs")
    assert H._lp_engine(None).name == "highs"
    assert H._lp_engine("highs").name == "highs"
    monkeypatch.setenv("DISCOPT_LP_STD_ENGINE", "rsut")
    with pytest.raises(ValueError, match="LP engine"):
        H._lp_engine(None)


@pytest.mark.parametrize("name", ["e226", "klein1", "forest6"])
def test_rust_engine_reaches_the_highs_verdict(name):
    sf = _instance(name)
    hi = H.solve_lp_std(sf, time_limit=60, engine="highs")
    ru = H.solve_lp_std(sf, time_limit=60, engine="rust")
    assert ru.status == hi.status, (ru.status, ru.message, hi.status)
    assert ru.gap_certified and hi.gap_certified
    if hi.status == "optimal":
        tol = 1e-6 * (1.0 + abs(hi.objective))
        assert abs(ru.objective - hi.objective) <= tol
        # Both bounds are rigorous: neither may exceed the other's objective.
        assert ru.bound <= hi.objective + tol and hi.bound <= ru.objective + tol
        assert ru.stats.get("lp/std_engine_rust") == 1.0
        assert "route/lp_milp_backend" not in ru.stats


def test_highs_engine_keeps_the_route_marker():
    out = H.solve_lp_std(_instance("e226"), time_limit=60, engine="highs")
    assert out.stats.get("route/lp_milp_backend") == 1.0
    assert out.labels["lp/bound_provenance"] == "ns-exact-dual-correction"
