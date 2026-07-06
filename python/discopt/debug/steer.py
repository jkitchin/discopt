"""Safe steering operations for the interactive debugger.

discopt is a global solver whose product is its correctness *certificate*
(CLAUDE.md §1). Unlike pounce, the debugger therefore refuses to mutate live
search state that could invalidate that certificate. The only levers exposed
here are ones the solver already trusts:

* ``inject`` -> ``PyTreeManager.inject_incumbent``: the tree re-validates the
  point (it is only accepted if genuinely feasible and improving), so a bad
  injection is rejected rather than believed.
* ``hint`` -> ``PyTreeManager.set_branch_hints``: reorders branching only; it
  cannot change which region is explored, only the order.

Neither can loosen a bound or fabricate an incumbent, so no debugger action can
turn a correct verdict into an incorrect one. Arbitrary node-box/bound edits
are intentionally *not* offered.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class DebugSteer:
    """Validated, certificate-safe steering handle over the Rust B&B tree."""

    def __init__(self, tree: Any, model: Any = None):
        self._tree = tree
        self._model = model

    def inject(self, solution: Any, obj: float) -> bool:
        """Offer a candidate incumbent; the tree accepts it only if valid.

        Parameters
        ----------
        solution : array-like
            Full primal vector in the solver's flat variable order.
        obj : float
            Objective value of ``solution``.

        Returns
        -------
        bool
            ``True`` if the tree adopted it as the new incumbent.
        """
        sol = np.asarray(solution, dtype=np.float64).ravel()
        if not np.all(np.isfinite(sol)):
            raise ValueError("inject: solution contains non-finite entries")
        if not np.isfinite(obj):
            raise ValueError("inject: objective is not finite")
        before = self._tree.incumbent()
        self._tree.inject_incumbent(sol, float(obj))
        after = self._tree.incumbent()
        # Adopted iff the incumbent objective strictly improved.
        if after is None:
            return False
        if before is None:
            return True
        return float(after[1]) < float(before[1]) - 1e-12

    def hint(self, node_ids: Any, var_indices: Any) -> None:
        """Suggest a branching variable per node (reorders only, never prunes).

        Parameters
        ----------
        node_ids : array-like of int
            Node ids (as exported by ``export_batch``).
        var_indices : array-like of int
            The variable index to branch on for each corresponding node.
        """
        ids = np.asarray(node_ids, dtype=np.int64).ravel()
        vhint = np.asarray(var_indices, dtype=np.int64).ravel()
        if ids.shape != vhint.shape:
            raise ValueError("hint: node_ids and var_indices must have equal length")
        self._tree.set_branch_hints(ids, vhint)
