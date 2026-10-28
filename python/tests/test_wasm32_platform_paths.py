"""Paths the wasm32/Pyodide build takes that a desktop run does not.

The browser page's ``examples`` job (``crates/discopt-wasm/tests/examples_in_pyodide.mjs``)
found two solver defects that only show on that platform:

* ``_MILP_ENGINE_UNCAPPED_NODES = 1 << 62`` is passed to the Rust engine as a
  ``usize``. On wasm32 that is 32-bit, so every MILP raised
  ``OverflowError: out of range integral type conversion attempted``.
* Pyodide has no OS threads, so MIQP node QPs take the serial POUNCE path, and no
  ``highspy``. The serial path computed its node lower bound with no multipliers,
  which falls through to the HiGHS tangent LP; without HiGHS it fell to the box-only
  bound, too weak to prune. The cardinality-portfolio example then stopped at
  ``feasible`` 0.04244 (13 nodes, bound -0.018) against a true optimum of 0.036667.

Both are reproduced here on desktop by simulating the platform.
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
import discopt.solver as S
import pytest

pytestmark = pytest.mark.smoke


def test_uncapped_node_count_fits_the_platform_usize():
    assert S._MILP_ENGINE_UNCAPPED_NODES <= sys.maxsize
    # Still effectively uncapped where usize is 64-bit (bound-neutral there).
    assert S._MILP_ENGINE_UNCAPPED_NODES >= min(1 << 62, sys.maxsize)
    assert S._MILP_ENGINE_UNCAPPED_NODES >= 2**31 - 1


def _cardinality_portfolio():
    """The page's MIQP example, verbatim in substance."""
    ret = [0.07, 0.11, 0.05, 0.14]
    cov = [
        [0.0400, 0.0120, 0.0020, 0.0180],
        [0.0120, 0.0900, 0.0030, 0.0400],
        [0.0020, 0.0030, 0.0100, 0.0040],
        [0.0180, 0.0400, 0.0040, 0.1600],
    ]
    m = dm.Model("portfolio with cardinality")
    w = m.continuous("w", shape=(4,), lb=0, ub=1)
    hold = m.binary("hold", shape=(4,))
    m.minimize(dm.sum([cov[i][j] * w[i] * w[j] for i in range(4) for j in range(4)]))
    m.subject_to(dm.sum([w[i] for i in range(4)]) == 1)
    m.subject_to(dm.sum([ret[i] * w[i] for i in range(4)]) >= 0.09)
    m.subject_to(dm.sum([hold[i] for i in range(4)]) <= 2)
    for i in range(4):
        m.subject_to(w[i] <= hold[i])
        m.subject_to(w[i] >= 0.10 * hold[i])
    return m


def test_serial_miqp_node_bound_certifies_without_highs(monkeypatch):
    """No threads and no HiGHS verifier, as under Pyodide: the serial node path
    must still bound tightly enough to certify. Exact optimum: assets 2 and 3 at
    5/9 and 4/9, variance 0.0366667."""
    pytest.importorskip("pounce")
    calls = {"serial": 0}
    real = S._convex_qp_node_lower_bound

    def counting(*args, **kwargs):
        if args[3] is None:  # the serial path passes G=None
            calls["serial"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(S, "_os_threads_available", lambda: False)
    monkeypatch.setattr(S, "_highs_lp_verifier_available", lambda: False)
    monkeypatch.setattr(S, "_convex_qp_node_lower_bound", counting)
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", "0")
    r = _cardinality_portfolio().solve(time_limit=60)
    assert calls["serial"] > 0, "the serial node path never ran; the test exercises nothing"
    assert r.status == "optimal"
    assert r.gap_certified
    assert r.objective == pytest.approx(0.0366667, abs=1e-6)
    assert r.bound <= r.objective + 1e-9
