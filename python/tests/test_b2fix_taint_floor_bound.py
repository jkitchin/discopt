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

_NVS05_OPT = 5.4709341  # MINLPLib reference optimum (minimize)


@pytest.mark.slow
def test_nvs05_reported_bound_is_sound_and_never_a_false_certificate():
    """nvs05's reported bound respects the per-node taint accounting invariants.

    HISTORY: this test originally pinned the exact taint-floor signature of the
    2026-07-10 tree (earliest taint at node 3 / iteration 2, floor
    1.3520892806701879). PF1 (#632, in-tree FBBT in the global spatial loop,
    2026-07-14) legitimately removed that early taint — every early node now
    certifies its own bound — so the pinned trajectory is falsified (measured
    2026-07-16: no sentinel fathom until the certification edge, iteration ~30;
    see docs/dev/nvs05-decline-taint-2026-07-16.md). What must SURVIVE any
    trajectory change is the accounting soundness B2-FIX (#89/#603) introduced:
    the reported bound is rigorous (never past the oracle, never past the
    incumbent), and a tree whose floor-inclusive gap has not closed never
    labels itself optimal (#27a). Those invariants are asserted here,
    trajectory-free. The exact-floor arithmetic stays unit-tested in
    test_spatial_cert_taint_floor_certify.py::
    test_gap_values_converged_matches_certification_arithmetic.
    """
    m = dm.from_nl(str(_DATA / "nvs05.nl"))
    r = m.solve(time_limit=20)

    # Soundness: a reported dual bound never crosses the oracle optimum.
    if r.bound is not None:
        assert math.isfinite(r.bound)
        assert r.bound <= _NVS05_OPT + 1e-3, f"bound {r.bound} crosses the optimum"
        # Certificate invariant: bound <= incumbent (minimize).
        if r.objective is not None:
            assert r.bound <= r.objective + 1e-9
    # A found incumbent is a genuine feasible value: never below the optimum.
    if r.objective is not None:
        assert r.objective >= _NVS05_OPT - 1e-6, (
            f"incumbent {r.objective} below the true optimum: infeasible point accepted"
        )
    # #27a contract: "optimal" is only legitimate with a genuinely closed gap.
    if r.status == "optimal":
        assert r.objective is not None and r.bound is not None
        assert abs(r.objective - _NVS05_OPT) < 1e-2
        gap = (r.objective - r.bound) / max(1.0, abs(r.objective))
        assert gap <= 1e-3, f"false certificate: status=optimal with gap {gap:.3g}"


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
