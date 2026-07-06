"""Safe steering operations for the interactive debugger.

discopt is a global solver whose product is its correctness *certificate*
(CLAUDE.md §1). Unlike pounce, the debugger therefore refuses to mutate live
search state that could invalidate that certificate. The only levers exposed
here are validated before they touch the tree:

* ``inject``: the debugger first validates the candidate against the
  **original** problem using the solver's own machinery (integrality check,
  integer rounding, constraint feasibility, true-objective evaluation via the
  ``validator`` wired in by the solve loop), and only then offers the point to
  ``PyTreeManager.inject_incumbent``, which additionally enforces strict
  improvement. ``inject_incumbent`` itself *trusts its caller* — it does NOT
  re-validate feasibility — so the validation step here is mandatory, and
  checkpoints that cannot wire a validator refuse to inject at all. The
  objective handed to the tree is always the evaluated objective at the point,
  never a relaxation bound.
* ``hint`` -> ``PyTreeManager.set_branch_hints``: reorders branching only; it
  cannot change which region is explored, only the order.

Neither can loosen a bound or fabricate an incumbent, so no debugger action can
turn a correct verdict into an incorrect one. Arbitrary node-box/bound edits
are intentionally *not* offered.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

#: Validator contract: ``validator(x) -> (feasible, x_validated, obj)`` where
#: ``x_validated`` has near-integral discrete coordinates snapped to integers
#: and ``obj`` is the true internal (min-sense) objective evaluated at the
#: point. ``obj`` is meaningless when ``feasible`` is False.
Validator = Callable[[np.ndarray], tuple[bool, np.ndarray, float]]


class DebugSteer:
    """Validated, certificate-safe steering handle over the Rust B&B tree."""

    def __init__(self, tree: Any, model: Any = None, validator: Optional[Validator] = None):
        self._tree = tree
        self._model = model
        self._validator = validator

    @property
    def can_inject(self) -> bool:
        """True when this checkpoint wired a candidate validator."""
        return self._validator is not None

    def inject(self, solution: Any) -> tuple[bool, Optional[float], str]:
        """Validate a candidate point and offer it as an incumbent.

        Parameters
        ----------
        solution : array-like
            Full primal vector in the solver's flat variable order.

        Returns
        -------
        (adopted, obj, reason) : tuple[bool, Optional[float], str]
            ``adopted`` is True iff the tree took the point as its new
            incumbent. ``obj`` is the validated true objective (internal min
            sense) when the point passed validation, else ``None``. ``reason``
            is a human-readable outcome.

        Raises
        ------
        ValueError
            If the candidate contains non-finite entries.
        RuntimeError
            If no validator is wired at this checkpoint — the point's true
            feasibility and objective cannot be verified, so injecting would
            risk fabricating an incumbent (refused, per CLAUDE.md §1/§3).
        """
        sol = np.asarray(solution, dtype=np.float64).ravel()
        if not np.all(np.isfinite(sol)):
            raise ValueError("inject: solution contains non-finite entries")
        if self._validator is None:
            raise RuntimeError(
                "inject: no candidate validator is wired at this checkpoint, so "
                "the point's feasibility and true objective cannot be verified; "
                "refusing to inject an unvalidated incumbent"
            )
        feasible, x_val, obj = self._validator(sol)
        if not feasible:
            return False, None, "rejected: infeasible for the original problem"
        obj = float(obj)
        adopted = bool(self._tree.inject_incumbent(np.ascontiguousarray(x_val), obj))
        if adopted:
            return True, obj, "adopted"
        return False, obj, "validated but not strictly improving on the incumbent"

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
