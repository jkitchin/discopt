"""Regression test: LP dual proofs on ``c513_lex_hpre`` under ``DISCOPT_MILP_CONFLICT``.

``data/c513_lex_hpre.mps.gz`` is the 513-element irreducible infeasible subsystem of
MIPLIB ``no-ip-65059`` (297 rows, 144 integer columns), with 7 lexicographic ordering
rows over its verified interchangeable block class, after HiGHS presolve. It is
infeasible. HiGHS 1.14 (1 thread, symmetry off) certifies it in 7,484 nodes.

The divergence this pins: ``farkas_proof_row`` (``bnb/milp_driver/highs_env.rs``)
returned no proof when the Farkas ray put weight on a row whose needed side is
infinite. HiGHS ``storeDualInfProof`` skips such a row instead (its ``continue`` on an
infinite row side). Measured at 60 s on the Rust engine: 18,697 of 39,689 LP-infeasible
nodes (47%) got no proof row. After the fix, 1 of 5,798 do. Node counts with the
conflict cut pool (#2347) to certify infeasibility: 128,736 before, 42,941 after.

The budget below sits between those two counts. It is a deterministic node budget, so
the verdict is a property of the search, not of the machine; ``time_limit`` is only a
backstop.
"""

from __future__ import annotations

from pathlib import Path

import discopt._rust as _rust
import discopt.modeling as dm
import pytest

FIXTURE = Path(__file__).parent / "data" / "c513_lex_hpre.mps.gz"
NODE_BUDGET = 70_000


@pytest.mark.slow
@pytest.mark.correctness
def test_c513_certified_infeasible_within_node_budget(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    monkeypatch.setenv("DISCOPT_MILP_CONFLICT", "1")
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    model = dm.from_mps(str(FIXTURE))
    _rust.profile_reset_py()
    r = model.solve(
        threads=1,
        max_nodes=NODE_BUDGET,
        deterministic=True,
        gap_tolerance=1e-4,
        time_limit=900,
    )
    ctr = _rust.profile_counters_py()
    # Prove the probe fired (CLAUDE.md §6): the conflict search ran, and its LP dual
    # proof step saw infeasible LPs.
    assert ctr.get("HighsMipSearchProofCalls", 0) > 0, ctr
    assert "native-milp" in r.algorithm_route, r.algorithm_route
    assert r.status == "infeasible", (r.status, int(r.node_count))
    assert r.node_count < NODE_BUDGET
    # Nearly every infeasible LP yields a proof row, as in HiGHS.
    assert ctr.get("HighsMipSearchProofNoRow", 0) * 100 <= ctr["HighsMipSearchProofCalls"], ctr
