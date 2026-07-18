"""Reconstruct a Tier-3 certificate tree from the Rust B&B recorder.

The Rust recorder (`PyTreeManager.tree_records()`) exports every node of the
branch-and-bound tree: id, parent, box (`lb`/`ub`), relaxation `local_lower_bound`,
and terminal `status`. This module turns that flat list into the nested tree the
exact-rational covering checker (`certificate.bnb.check_tree_covers`) consumes,
**deriving each branch's split variable and point from the parent-vs-child box
difference** — the recorder itself stores no branch metadata, keeping it
bound-neutral.

A branch node's two children each equal the parent box except in one column `j`
(the branch variable): one child tightens the upper bound to the split point, the
other tightens the lower bound. That is exactly the invariant `check_tree_covers`
verifies, so a faithfully-recorded tree reconstructs into one that covers the root.
"""

from __future__ import annotations

from fractions import Fraction

from .bnb import check_tree_covers
from .schema import to_rational


def _rat_box(arr) -> list:
    return [to_rational(float(v)) for v in arr]


def records_to_tree(records: list[dict]) -> tuple[dict, object]:
    """Convert flat ``tree_records()`` output to ``(tree, root_id)`` for covering.

    ``tree`` maps id -> ``{"box", "kind", "branch"}`` where a node with children is
    a ``branch`` (its split var/point derived from the child boxes) and a childless
    node is a ``leaf``. Raises ``ValueError`` on a malformed tree (a branch without
    exactly two children, or children that do not differ from the parent in exactly
    one column).
    """
    by_id = {r["id"]: r for r in records}
    children: dict[object, list] = {}
    root_id = None
    for r in records:
        p = r["parent"]
        if p is None:
            root_id = r["id"]
        else:
            children.setdefault(p, []).append(r["id"])
    if root_id is None:
        raise ValueError("no root node (a node with parent=None) in records")

    tree: dict = {}
    for r in records:
        nid = r["id"]
        box = {"lb": _rat_box(r["lb"]), "ub": _rat_box(r["ub"])}
        kids = children.get(nid, [])
        if not kids:
            tree[nid] = {"box": box, "kind": "leaf"}
            continue
        if len(kids) != 2:
            raise ValueError(f"node {nid} has {len(kids)} children (expected 2 for a binary split)")
        var, point, ordered = _derive_split(r, by_id[kids[0]], by_id[kids[1]])
        tree[nid] = {
            "box": box,
            "kind": "branch",
            "branch": {"var": var, "point": point, "children": ordered},
        }
    return tree, root_id


def _derive_split(parent: dict, c1: dict, c2: dict) -> tuple[int, list, list]:
    """Find the branch column, the split point, and order children [left, right]."""
    plb = [Fraction(float(v)) for v in parent["lb"]]
    pub = [Fraction(float(v)) for v in parent["ub"]]
    # The branch column is where a child's box differs from the parent's.
    diff_cols = set()
    for child in (c1, c2):
        clb = [Fraction(float(v)) for v in child["lb"]]
        cub = [Fraction(float(v)) for v in child["ub"]]
        for j in range(len(plb)):
            if clb[j] != plb[j] or cub[j] != pub[j]:
                diff_cols.add(j)
    if len(diff_cols) != 1:
        raise ValueError(f"children differ from parent in {len(diff_cols)} columns (expected 1)")
    j = diff_cols.pop()

    # Left child tightens the upper bound (ub[j] < parent ub[j]); right tightens lb.
    def is_left(child):
        return Fraction(float(child["ub"][j])) < pub[j]

    left, right = (c1, c2) if is_left(c1) else (c2, c1)
    left_ub = Fraction(float(left["ub"][j]))
    right_lb = Fraction(float(right["lb"][j]))
    # Spatial split: left.ub == right.lb == point. Integer split: left.ub=floor,
    # right.lb=floor+1 -> any non-integer point in between works; use the midpoint
    # so floor(point) == left.ub (what check_tree_covers recomputes).
    point = left_ub if left_ub == right_lb else left_ub + Fraction(1, 2)
    return j, [point.numerator, point.denominator], [left["id"], right["id"]]


def check_recorded_tree_covers(records: list[dict], integer_cols: set[int]) -> tuple[bool, str]:
    """Reconstruct the tree from ``records`` and verify it covers the root box."""
    tree, root_id = records_to_tree(records)
    return check_tree_covers(tree, root_id, integer_cols)
