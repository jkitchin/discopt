"""#1270: an empty B&B tree certifies only if nothing was removed unproven.

The NLP-BB, MIQP-BB and spatial exits granted ``optimal`` on
``_gap_converged(...) or tree.is_finished()``. The Rust tree finishes when no
node is open, including after an untrusted node was fathomed with no branch
direction. With a finite inherited bound that fathom seeds
``unresolved_floor`` (#598) rather than ``bound_unresolved``, and only the
latter was checked. Measured before the fix: seeded ``tls2`` with one stalled
convex node (``DISCOPT_CONVEX_STALL_ABSTAIN`` default arm) returned
``optimal``/``gap_certified=True`` at 5.3 with a bound of 2.81, a 47% gap and no
``gap_criterion``.
"""

from __future__ import annotations

import pytest
from discopt.solver import _tree_exhausted_with_proof

INF = float("inf")


class _Tree:
    def __init__(self, finished, bound_unresolved=False, unresolved_floor=INF):
        self._finished = finished
        self._stats = {"bound_unresolved": bound_unresolved, "unresolved_floor": unresolved_floor}

    def is_finished(self):
        return self._finished

    def stats(self):
        return self._stats


@pytest.mark.unit
@pytest.mark.parametrize(
    "tree, expected",
    [
        (_Tree(True), True),  # exhaustive search, every removal proved
        (_Tree(False), False),  # open nodes remain
        (_Tree(True, bound_unresolved=True), False),  # #467 -inf pin
        (_Tree(True, unresolved_floor=2.81), False),  # #598 finite floor
        (_Tree(True, unresolved_floor=-INF), False),
    ],
)
def test_empty_tree_is_a_proof_only_without_unproven_removals(tree, expected):
    assert _tree_exhausted_with_proof(tree) is expected
