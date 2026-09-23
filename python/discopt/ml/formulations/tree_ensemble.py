"""MILP formulation for tree ensemble models (Misic 2020).

For each tree with L leaves, the encoding introduces L binary variables
(exactly one active) and big-M constraints linking input features to the
selected leaf via split decisions along the root-to-leaf path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import discopt.modeling as dm
from discopt.ml.scaling import OffsetScaling
from discopt.ml.tree import TreeEnsembleDefinition

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable


class TreeEnsembleFormulation:
    """Embed a tree ensemble as MILP constraints.

    Parameters
    ----------
    model : discopt.Model
        Optimization model to add variables and constraints to.
    ensemble : TreeEnsembleDefinition
        The trained tree ensemble.
    prefix : str
        Name prefix for all created variables and constraints.
    scaling : OffsetScaling, optional
        Input/output affine scaling.
    split_eps : float
        Half-width of the separation band around each split threshold, applied
        **symmetrically**: a leaf whose path goes left at a node requires
        ``x[j] <= thr - split_eps``, and one that goes right requires
        ``x[j] >= thr + split_eps`` (#1447).

        It must exceed the solver's absolute feasibility tolerance (1e-6). A
        MILP solver may satisfy a constraint to within that tolerance, and a
        decision tree is **discontinuous** at its thresholds, so tolerance-sized
        slack becomes an O(1) error in the prediction. Before #1447 only the
        right branch was separated, and by exactly 1e-6 -- equal to the
        tolerance, so its protection was fully consumed. The left branch had
        none at all, which is how a returned point could sit 3.1e-14 on the
        *wrong* side of a threshold while the binary claimed the leaf on the
        near side: the MIP reported 3.1526 where ``ensemble.predict(x)`` gave
        -2.8501, a discrepancy of 6.00 at the optimizer's own solution.

        The margin that matters is ``split_eps - tol``, which must be positive:
        a leaf's rows are satisfiable to within ``tol``, so at ``split_eps ==
        tol`` the protection is exactly consumed. Hence the 1e-5 default, ten
        times the tolerance. **Measured** on a one-node tree splitting ``[0, 1]``
        at 0.5, for both 1e-6 and 1e-5, the set the encoding excludes is *not* the
        full ``(thr - eps, thr + eps)`` band -- ``thr - eps`` and ``thr + eps``
        both solve, and so does ``thr - tol`` -- it is the threshold **point**
        itself:

        ===============  ==================  ===========
        ``x``            ``predict(x)``      MILP
        ===============  ==================  ===========
        ``0.5 - eps``    1.0                 1.0
        ``0.5``          1.0                 **infeasible**
        ``0.5 + eps``    3.0                 3.0
        ===============  ==================  ===========

        That is the trade, stated plainly: a point sitting *exactly* on a
        threshold now makes the model infeasible, where before it was assigned a
        leaf it may not belong to. A discontinuous function cannot have both --
        one side of the breakpoint must be given up -- and giving it up **loudly**
        beats assigning it silently to the leaf with the better value, which is
        the direction that admits values the tree never attains. A tree is
        piecewise constant, so this costs no attainable leaf value in an
        unconstrained optimization; it can only bite a model whose other
        constraints pin an input onto a threshold, and then it says so.
    """

    def __init__(
        self,
        model: Model,
        ensemble: TreeEnsembleDefinition,
        prefix: str,
        scaling: OffsetScaling | None = None,
        split_eps: float = 1e-5,
    ):
        if ensemble.input_bounds is None:
            raise ValueError("TreeEnsembleDefinition.input_bounds is required for MILP formulation")
        self._model = model
        self._ensemble = ensemble
        self._prefix = prefix
        self._scaling = scaling
        self._split_eps = split_eps

    def build(self) -> tuple[Variable, Variable]:
        """Create all variables and constraints.

        Returns
        -------
        inputs : Variable
            Input feature variables, shape ``(n_features,)``.
        outputs : Variable
            Ensemble output variable, shape ``(1,)``.
        """
        m = self._model
        ens = self._ensemble
        pfx = self._prefix
        assert ens.input_bounds is not None  # validated in __init__
        lb, ub = ens.input_bounds

        inputs = m.continuous(
            f"{pfx}_input",
            shape=(ens.n_features,),
            lb=lb,
            ub=ub,
        )

        lb_arr = np.asarray(lb, dtype=np.float64)
        ub_arr = np.asarray(ub, dtype=np.float64)

        tree_output_exprs = []
        out_lb = float(ens.base_score)
        out_ub = float(ens.base_score)
        for t, tree in enumerate(ens.trees):
            leaves = tree.leaves
            n_leaves = len(leaves)

            # Binary: exactly one leaf selected
            z = m.binary(f"{pfx}_t{t}_leaf", shape=(n_leaves,))
            m.subject_to(
                dm.sum(lambda k: z[k], over=range(n_leaves)) == 1,
                name=f"{pfx}_t{t}_one_leaf",
            )

            # Split constraints for each leaf's root-to-leaf path
            for l_idx, leaf in enumerate(leaves):
                for node, direction in tree.leaf_ancestors(leaf):
                    j = int(tree.feature[node])
                    thr = float(tree.threshold[node])

                    if direction == "left":
                        # x[j] <= threshold - eps when this leaf is selected.
                        # The `- eps` is #1447: without it the constraint is
                        # `x <= thr`, which a solver may satisfy to within its
                        # feasibility tolerance, letting x sit just ABOVE thr
                        # while this leaf is claimed -- the side `predict()`
                        # sends to the sibling. Separated symmetrically with the
                        # right branch below, so a claimed leaf really does
                        # contain the returned point.
                        #
                        # The per-constraint big-M `max(ub_j - rhs, 0)` is
                        # exactly the slack needed to reach the feature's upper
                        # bound when z=0, and inert (clamped to 0) for out-of-box
                        # thresholds, so a non-selected leaf's row never cuts a
                        # feasible point (F2).
                        rhs_thr = thr - self._split_eps
                        M_j = max(float(ub_arr[j]) - rhs_thr, 0.0)
                        m.subject_to(
                            inputs[j] <= rhs_thr + M_j * (1 - z[l_idx]),
                            name=f"{pfx}_t{t}_sL_{node}_{l_idx}",
                        )
                    else:
                        # x[j] >= threshold + eps when this leaf is selected.
                        # Big-M `max(thr + eps - lb_j, 0)` reaches the feature's
                        # lower bound when z=0 and clamps to 0 for out-of-box
                        # thresholds (F2).
                        rhs_thr = thr + self._split_eps
                        M_j = max(rhs_thr - float(lb_arr[j]), 0.0)
                        m.subject_to(
                            inputs[j] >= rhs_thr - M_j * (1 - z[l_idx]),
                            name=f"{pfx}_t{t}_sR_{node}_{l_idx}",
                        )

            # Tree output: sum of z[l] * leaf_value[l]
            leaf_vals = np.array(
                [float(tree.value[leaf]) for leaf in leaves],
                dtype=np.float64,
            )
            y_t = dm.sum(
                lambda k, _v=leaf_vals: z[k] * float(_v[k]),
                over=range(n_leaves),
            )
            tree_output_exprs.append(y_t)
            # Free output bounds (T-N0.4): exactly one leaf fires per tree, so
            # each tree contributes a value in [min leaf, max leaf].
            out_lb += float(leaf_vals.min())
            out_ub += float(leaf_vals.max())

        # Ensemble output: sum of trees + base_score
        outputs = m.continuous(f"{pfx}_output", shape=(1,), lb=out_lb, ub=out_ub)
        total = tree_output_exprs[0]
        for expr in tree_output_exprs[1:]:
            total = total + expr
        if ens.base_score != 0.0:
            total = total + ens.base_score
        m.subject_to(outputs[0] == total, name=f"{pfx}_ensemble_out")

        return inputs, outputs
