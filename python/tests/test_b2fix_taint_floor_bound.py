"""B2-FIX (task #89): tainted trees report the strongest RIGOROUS dual bound.

DECOMP-1 (docs/dev/certification-effort-decomposition-2026-07-10.md §3, Lever B)
measured that a single ``_nonrigorous_sentinel_fathom`` taint made the result
build discard the ENTIRE tree bound and fall back to the much weaker root
bound (nvs05: reported 1.3481 while the frontier had reached 4.87).

The sound repair is per-node accounting, not taint removal: a node fathomed via
the failure sentinel *without* an infeasibility proof was removed from the
frontier unsoundly, but its POP-TIME bound (inherited from its parent's rigorous
relaxation solve — the Rust import floors every child at the parent bound) is
still a valid lower bound for its whole subtree. The rigorous global dual bound
of a tainted tree is therefore::

    min(surviving-frontier bound, min over tainted nodes of pop-time bound)

On nvs05 the earliest taint (node 3, iteration 2) carries pop-time bound
1.35209, so the honest rigorous bound is ~1.3521 — NOT the 4.87 frontier value
(that removed subtree was never re-proven; DECOMP-1's "the tree proved 4.8746"
is falsified by this per-node accounting). The regression under test is that
the reported bound is exactly this floored value (strictly stronger than the
pre-fix 1.3481 root fallback), stays on the sound side of the oracle optimum
5.4709, and the taint still blocks an "optimal" upgrade (issue #27a contract).
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import PyTreeManager

_DATA = Path(__file__).parent / "data" / "minlplib"

_NVS05_OPT = 5.4709  # MINLPLib reference optimum (minimize)
# Pre-fix reported bound: root-relaxation fallback after the tree bound was
# discarded wholesale (decertify-and-discard).
_NVS05_PREFIX_BOUND = 1.3481188467674154
# Post-fix rigorous bound: the taint floor — the pop-time bound of the earliest
# non-rigorously fathomed node (node 3, iteration 2), which caps the reported
# bound for the rest of the run. Deterministic: it derives from the root
# FBBT/OBBT box and the first two batch iterations, not from the time limit.
_NVS05_TAINT_FLOOR = 1.3520892806701879


@pytest.mark.slow
def test_nvs05_tainted_tree_reports_rigorous_floored_bound():
    """nvs05's tainted exit reports the floored rigorous bound, not the weak
    root fallback (fails before B2-FIX: reported 1.3481 < floor 1.3521)."""
    m = dm.from_nl(str(_DATA / "nvs05.nl"))
    # The earliest taint fires by iteration 2 (~4 s in); 20 s leaves margin
    # without paying the full 60 s DECOMP-1 budget.
    r = m.solve(time_limit=20)

    assert r.objective is not None and abs(r.objective - _NVS05_OPT) < 1e-2
    assert r.bound is not None and math.isfinite(r.bound)
    # Soundness: never report past the oracle optimum (minimize).
    assert r.bound <= _NVS05_OPT + 1e-3, f"bound {r.bound} crosses the optimum"
    # Certificate invariant: bound <= incumbent.
    assert r.bound <= r.objective + 1e-9
    # The regression: the reported bound must be the rigorous taint-floored
    # value, strictly stronger than the pre-fix discarded-tree fallback.
    assert r.bound >= _NVS05_TAINT_FLOOR - 1e-6, (
        f"reported bound {r.bound} is weaker than the rigorous taint floor "
        f"{_NVS05_TAINT_FLOOR} (decertify-and-discard regression)"
    )
    # ... and it must NEVER exceed what the taint floor allows: the frontier
    # value (~4.87 at 60 s) is NOT rigorous once a subtree floored at 1.3521
    # was removed without proof. Reporting materially past the floor would be
    # a false certificate.
    assert r.bound <= _NVS05_TAINT_FLOOR + 1e-6, (
        f"reported bound {r.bound} exceeds the rigorous taint floor "
        f"{_NVS05_TAINT_FLOOR}: an unsoundly-fathomed subtree's bound was dropped"
    )
    # #27a contract: a tainted tree never upgrades to "optimal" via its own
    # recovered bound.
    assert r.status != "optimal"


@pytest.mark.smoke
def test_node_lower_bounds_returns_pop_time_bounds():
    """The Rust tree exposes pop-time node bounds; unknown ids map to -inf."""
    t = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
    t.initialize()
    lb, ub, ids, _psols = t.export_batch(4)
    assert len(ids) == 1  # root only
    pop = np.asarray(t.node_lower_bounds(np.asarray(ids, dtype=np.int64)))
    # Root has no parent bound: -inf (a -inf floor keeps the conservative
    # discard behavior in the driver).
    assert pop[0] == -np.inf

    # Import a finite bound and branch; children inherit >= the parent bound.
    t.import_results(
        np.asarray(ids, dtype=np.int64),
        np.array([3.5]),
        np.array([[0.5, 0.5]]),
        np.array([False]),
    )
    t.process_evaluated()
    lb2, ub2, ids2, _ = t.export_batch(4)
    assert len(ids2) >= 1
    pop2 = np.asarray(t.node_lower_bounds(np.asarray(ids2, dtype=np.int64)))
    assert np.all(pop2 >= 3.5 - 1e-12), f"children lost the parent floor: {pop2}"

    # Out-of-range / negative ids are conservative (-inf), never an exception.
    bad = np.asarray(t.node_lower_bounds(np.array([999, -1], dtype=np.int64)))
    assert bad[0] == -np.inf and bad[1] == -np.inf
