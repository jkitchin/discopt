"""#2115: the in-house MILP driver must bound nodes of a translation-invariant master.

The MINLPLib ``m6`` final OA master (``data/oa_masters/m6_final_master_2115.npz``,
captured from a default ``Model.solve()`` with ``pyscipopt`` absent, i.e. exactly
what ``milp_highs._run_simplex_engine`` receives for the #2069 independent
confirmation) has 86 columns, 30 binary, 258 rows, and **42 columns free on both
sides**: layout positions, so the problem is translation-invariant and no bound on
them exists.

Before #2115 the driver's node bound -- the Neumaier-Shcherbina safe bound from the
node's dual -- abstained on 3194 of ~29k nodes (``NodeNsAbstain``): a free or
one-sided-open column whose reduced cost is zero in exact arithmetic is only known
up to rounding, which gives the bound ``-inf``. Abstaining nodes with nothing left
to branch are removed unbounded and floor the tree at 62.83 against an optimum of
82.2568769; the tree drains in ~38.6k nodes and the engine reports
``iteration_limit``, so ``m6`` cannot be certified without SCIP.

``DISCOPT_NS_PROJECT`` (default ON since its #2115 graduation panel; ``=0`` opts
out) adds an exact dual projection as the last rung of the NS ladder
(``crates/discopt-core/src/lp/simplex/project.rs``): the open columns' reduced
costs are made exactly zero on an enclosure of a nearby dual, and the bound is
evaluated rigorously over the enclosure. Measured here at #2115: abstentions
3194 -> 0, master certified optimal (2026-11-17, #2226: 219 -> 3 on a driver that
now needs ~1.1k nodes; see the tests). The Rust unit tests in ``project.rs`` pin the mechanism
on a 2x4 LP (including a 2000-dual soundness sweep); this pins the real master.

Each arm runs in its own process: the flag is read once per process (a Rust
``OnceLock``), so an in-process A/B would silently run one arm twice.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

DATA = os.path.join(os.path.dirname(__file__), "data")
MASTER = os.path.join(DATA, "oa_masters", "m6_final_master_2115.npz")

#: MINLPLib reference optimum for m6 (``minlplib.solu``). The master's optimum is
#: the MINLP's: this is the final OA master, whose claim SCIP confirms.
M6_OPTIMUM = 82.2568769100

_RUNNER = r"""
import json, sys
import numpy as np, scipy.sparse as sp
from discopt.solvers import milp_simplex
from discopt._rust import profile_counters_py, profile_reset_py

d = np.load(sys.argv[1])
A = sp.csr_matrix((d["A_ub_data"], d["A_ub_indices"], d["A_ub_indptr"]),
                  shape=tuple(d["A_ub_shape"]))
profile_reset_py()
r = milp_simplex.solve_milp(
    c=d["c"], A_ub=A, b_ub=d["b_ub"], bounds=list(zip(d["lb"], d["ub"])),
    integrality=d["integrality"], time_limit=float(sys.argv[2]),
    gap_tolerance=float(d["gap_tolerance"]), mip_start=d["mip_start"],
)
c = dict(profile_counters_py())
print("RESULT " + json.dumps({
    "status": r.status.value, "bound": r.bound, "objective": r.objective,
    "nodes": int(r.node_count), "abstain": int(c.get("NodeNsAbstain", 0)),
    "rescue": int(c.get("NodeNsProjectRescue", 0)),
    "ns_bound": int(c.get("NodeNsBound", 0)),
}), flush=True)
"""


def _solve(flag: str | None, time_limit: float) -> dict:
    """``flag=None`` leaves ``DISCOPT_NS_PROJECT`` unset: the shipped default."""
    env = {**os.environ, "DISCOPT_PROFILE": "1"}
    env.pop("DISCOPT_NS_PROJECT", None)
    if flag is not None:
        env["DISCOPT_NS_PROJECT"] = flag
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, MASTER, str(time_limit)],
        capture_output=True,
        text=True,
        env=env,
        timeout=time_limit * 4 + 120,
    )
    for ln in proc.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[len("RESULT ") :])
    raise AssertionError(f"no result (rc={proc.returncode}): {proc.stderr[-800:]}")


@pytest.mark.slow
@pytest.mark.correctness
def test_m6_master_certifies_with_ns_projection():
    """The shipped default (flag unset, graduated ON in #2115): optimal, the bound
    within the master's gap of the optimum and never above it, and the projection
    is what supplied the bounds (anti-vacuity, §6).

    Measured at #2115: 25 406 nodes, 9.1 s, abstentions 3194 -> 0. Re-measured
    2026-11-17 (#2226; the driver since #2196 solves the master in ~1.1k nodes,
    deterministic over 3 runs): ON 1072 nodes, 3 abstentions, 119 rescues; OFF 1212
    nodes, 219 abstentions. The bar is therefore relative to the OFF arm of the same
    build, not an absolute count: the projection removes at least 90% of the
    abstentions. The 120 s limit is a backstop, not the bar.
    """
    r = _solve(None, 120.0)
    off = _solve("0", 120.0)
    # The NS ladder ran, and its last rung is what fired.
    assert r["ns_bound"] > 0, r
    assert r["rescue"] > 0, r
    assert off["abstain"] > 0, off  # the defect the projection addresses is present
    assert 10 * r["abstain"] <= off["abstain"], (r, off)
    assert r["status"] == "optimal", r
    tol = 1e-6 * (1.0 + M6_OPTIMUM)
    assert r["bound"] <= M6_OPTIMUM + tol, f"bound {r['bound']} above the optimum"
    assert r["bound"] >= M6_OPTIMUM * (1.0 - 1e-4) - tol, r
    assert r["objective"] >= M6_OPTIMUM - tol, r


@pytest.mark.slow
@pytest.mark.correctness
def test_m6_master_premise_without_ns_projection():
    """OFF (``DISCOPT_NS_PROJECT=0``, premise): the plain NS ladder abstains on
    nodes the projection would rescue, and no rescue happens. If the abstentions
    vanish, the defect moved and the ON test above no longer exercises the
    projection.

    Measured at #2115: 38 594 nodes, ``iteration_limit`` at bound 62.83 after 14 s.
    Re-measured 2026-11-17 (#2226): 1212 nodes, 219 abstentions, and the master now
    certifies ``optimal`` without the projection -- the abstaining nodes no longer
    floor a tree this small -- so the premise is the abstentions themselves, not a
    failed certificate. Any certificate it does give must still be sound.
    """
    r = _solve("0", 120.0)
    assert r["rescue"] == 0, r
    assert r["abstain"] > 0, r
    tol = 1e-6 * (1.0 + M6_OPTIMUM)
    assert r["bound"] is None or r["bound"] <= M6_OPTIMUM + tol, r
    if r["status"] == "optimal":
        assert r["objective"] >= M6_OPTIMUM - tol, r
