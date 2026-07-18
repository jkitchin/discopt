"""Rust B&B recorder -> Tier-3 tree reconstruction -> covering check.

Exercises the recorder binding (`PyTreeManager.tree_records()`) end-to-end: drive a
tiny branch-and-bound tree, export it, reconstruct the nested tree (deriving the
branch split from parent/child box differences), and verify it covers the root
box. No real solve is involved, so these are ``smoke``. They require the built Rust
extension.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.certificate.bnb import check_tree_covers
from discopt.certificate.bnb_record import check_recorded_tree_covers, records_to_tree

PyTreeManager = pytest.importorskip("discopt._rust").PyTreeManager


def _branch_once():
    """Root [0,1]^2 with a fractional integer var 0 -> one integer branch."""
    tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
    tm.initialize()
    _lb, _ub, ids, _ps = tm.export_batch(1)
    tm.import_results(
        np.array(ids, dtype=np.int64),
        np.array([1.0]),
        np.array([[0.5, 0.0]]),  # var 0 fractional
        np.array([False]),  # not integer-feasible -> branch
    )
    tm.process_evaluated()
    return tm


@pytest.mark.smoke
def test_recorder_exports_root_only():
    tm = PyTreeManager(2, [0.0, 0.0], [1.0, 1.0], [0], [2], "best_first")
    tm.initialize()
    recs = tm.tree_records()
    assert len(recs) == 1
    assert recs[0]["parent"] is None
    assert list(recs[0]["lb"]) == [0.0, 0.0]
    assert list(recs[0]["ub"]) == [1.0, 1.0]


@pytest.mark.smoke
def test_recorder_branch_reconstructs_and_covers():
    tm = _branch_once()
    recs = tm.tree_records()
    assert len(recs) == 3  # root + two children
    tree, root = records_to_tree(recs)
    assert tree[root]["kind"] == "branch"
    assert tree[root]["branch"]["var"] == 0
    # Both children are leaves that split column 0.
    assert sum(1 for n in tree.values() if n["kind"] == "leaf") == 2
    ok, reason = check_recorded_tree_covers(recs, integer_cols={0, 1})
    assert ok, reason


@pytest.mark.smoke
def test_recorded_status_strings():
    recs = _branch_once().tree_records()
    by_id = {r["id"]: r for r in recs}
    assert by_id[0]["status"] == "branched"
    assert by_id[1]["status"] in ("pending", "pruned", "fathomed")


@pytest.mark.smoke
def test_reconstruction_rejects_tampered_gap():
    # Hand-built records whose children leave a gap in the root box: reconstruction
    # + covering must reject (the split point is not shared).
    tampered = [
        {
            "id": 0,
            "parent": None,
            "depth": 0,
            "lb": [0.0],
            "ub": [2.0],
            "local_lower_bound": 0.0,
            "status": "branched",
        },
        {
            "id": 1,
            "parent": 0,
            "depth": 1,
            "lb": [0.0],
            "ub": [0.5],
            "local_lower_bound": 0.0,
            "status": "pruned",
        },
        {
            "id": 2,
            "parent": 0,
            "depth": 1,
            "lb": [1.0],
            "ub": [2.0],  # gap (0.5, 1.0)
            "local_lower_bound": 0.0,
            "status": "pruned",
        },
    ]
    ok, reason = check_recorded_tree_covers(tampered, integer_cols=set())
    assert not ok


@pytest.mark.smoke
def test_spatial_two_level_tree_covers():
    # A hand-built 2-level spatial tree over [0,4]: split at 2, then left split at 1.
    recs = [
        {
            "id": 0,
            "parent": None,
            "depth": 0,
            "lb": [0.0],
            "ub": [4.0],
            "local_lower_bound": 0.0,
            "status": "branched",
        },
        {
            "id": 1,
            "parent": 0,
            "depth": 1,
            "lb": [0.0],
            "ub": [2.0],
            "local_lower_bound": 0.0,
            "status": "branched",
        },
        {
            "id": 2,
            "parent": 0,
            "depth": 1,
            "lb": [2.0],
            "ub": [4.0],
            "local_lower_bound": 0.0,
            "status": "pruned",
        },
        {
            "id": 3,
            "parent": 1,
            "depth": 2,
            "lb": [0.0],
            "ub": [1.0],
            "local_lower_bound": 0.0,
            "status": "pruned",
        },
        {
            "id": 4,
            "parent": 1,
            "depth": 2,
            "lb": [1.0],
            "ub": [2.0],
            "local_lower_bound": 0.0,
            "status": "pruned",
        },
    ]
    tree, root = records_to_tree(recs)
    assert check_tree_covers(tree, root, set())[0]
