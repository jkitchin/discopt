"""Convexity-aware presolve pass (D1 of issue #53).

Promotes the box-local convexity certificate (``discopt._jax.convexity``)
from a B&B-time tool into a presolve pass that runs inside the
orchestrator's fixed-point loop. The pass scans every constraint body,
asks the certificate whether the body is convex (or concave) on the
current variable box, and stamps the indices of provably-convex
constraints into the delta's ``StructureManifest``.

In v0 the pass is *informational*: downstream consumers (the relaxation
compiler, B&B initialisation) can skip building a McCormick relaxation
for a convex inequality `f(x) ≤ rhs` because the original constraint is
already its own convex relaxation. Future work (per the D1 roadmap
entry) extends this into structural rewrites — epigraph substitution
for non-smooth convex atoms, exp-cone for log-sum-exp, second-order-cone
for PSD-quadratics — by reusing the same scan to identify candidates.

The certificate is sound: it never claims convexity for a non-convex
body, so attaching the marker is always safe. Calls that fail (array
variables, transcendentals outside the certificate's atom set,
indefinite Hessians) abstain and are silently skipped.

References
----------
- Boyd, Vandenberghe (2004), *Convex Optimization*, Ch. 4.
- Lubin, Yamangil, Bent, Vielma (2018), *Polyhedral approximation in
  mixed-integer convex optimization*, Math. Prog. 172.

Dependencies: A3 (Python passes participate in the orchestrator via
``run_orchestrated_presolve``).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .protocol import make_python_delta


class ConvexReformPass:
    """Detect convex constraints and mark them in the structural manifest.

    The pass operates on the *Python* ``Model`` object — not the Rust
    ``PyModelRepr`` — because the box-local certificate
    (`discopt._jax.convexity.certificate.certify_convex`) consumes the
    Python expression DAG. The pass therefore has to be invoked via
    `run_orchestrated_presolve` with the Python model bound through a
    closure, e.g.::

        from discopt._jax.presolve import run_orchestrated_presolve
        from discopt._jax.presolve.convex_reform import ConvexReformPass

        new_repr, stats = run_orchestrated_presolve(
            model_repr,
            python_passes=[ConvexReformPass(model)],
        )

    Args:
        model: the Python ``Model`` whose constraints to analyse.
        box: optional ``{Variable: Interval}`` overriding the model's
            declared bounds. Useful for B&B-node-local presolve where
            tightening from FBBT is already in hand. ``None`` falls
            back to the model's declared variable bounds.
    """

    name = "convex_reform"

    def __init__(self, model: Any, box: Optional[dict] = None):
        self.model = model
        self.box = box

    def run(self, model_repr: Any) -> dict:
        from discopt._jax.convexity.certificate import certify_convex
        from discopt._jax.convexity.lattice import Curvature

        delta = make_python_delta(self.name)
        # Refresh the box from the (possibly tightened) model_repr if
        # the caller didn't supply one. We pull lb/ub element-wise from
        # the Rust IR — since A3's `tighten_var_bounds` writes back any
        # Python pass tightening, this stays in sync sweep-to-sweep.
        box = self.box
        if box is None:
            box = _box_from_model_repr(self.model, model_repr)

        convex_indices: list[int] = []
        examined = 0
        for ci, c in enumerate(self.model._constraints):
            body = getattr(c, "body", None)
            if body is None:
                continue
            examined += 1
            try:
                verdict = certify_convex(body, self.model, box=box)
            except Exception:
                # Certificate threw — abstain. The pass must never
                # propagate exceptions because the orchestrator catches
                # them and stamps an "error" delta, which would
                # incorrectly imply the model is unprocessable.
                continue
            sense = getattr(c, "sense", None)
            # `f(x) ≤ rhs` is a convex feasible set when f is convex.
            # `f(x) ≥ rhs` is a convex feasible set when f is concave
            # (since `-f ≤ -rhs` is convex). Equality is convex only
            # for affine bodies; the certificate cannot prove that
            # without help from the linear walker.
            if sense == "<=" and verdict == Curvature.CONVEX:
                convex_indices.append(ci)
            elif sense == ">=" and verdict == Curvature.CONCAVE:
                convex_indices.append(ci)
        delta["work_units"] = examined
        # Stamp into the structural manifest. The pass is diagnostic —
        # it does not modify the model — so this does NOT count as
        # `delta_made_progress`, matching how cliques and the structural
        # manifest are treated in `delta_made_progress`.
        delta["convex_constraints"] = convex_indices
        return delta


def _box_from_model_repr(model: Any, model_repr: Any) -> dict:
    """Build a ``{Variable: Interval}`` box from the Rust IR's bounds.

    The Rust IR may carry tighter bounds than the Python ``Model``
    (FBBT and other Rust-side passes write into ``model_repr.var_lb /
    var_ub`` first). Reading from the IR keeps the certificate's box
    in sync with whatever upstream presolve passes have produced.
    """
    from discopt._jax.convexity.interval import Interval

    box: dict = {}
    py_blocks = list(model._variables)
    n = min(len(py_blocks), model_repr.n_var_blocks)
    for bi in range(n):
        v = py_blocks[bi]
        rust_lb = np.asarray(model_repr.var_lb(bi), dtype=np.float64)
        rust_ub = np.asarray(model_repr.var_ub(bi), dtype=np.float64)
        py_lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        py_ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        if rust_lb.size != py_lb.size or rust_ub.size != py_ub.size:
            # Aux variables introduced by polynomial reformulation have
            # no Python-side counterpart; skip them.
            continue
        # Intersection (the Rust IR may have tightened by element).
        eff_lb = np.maximum(py_lb, rust_lb)
        eff_ub = np.minimum(py_ub, rust_ub)
        if eff_lb.size == 1:
            box[v] = Interval(
                np.asarray(eff_lb[0], dtype=np.float64),
                np.asarray(eff_ub[0], dtype=np.float64),
            )
        else:
            # Vector / matrix bounds — the certificate accepts these
            # via element-wise Interval; pass the array form.
            box[v] = Interval(
                eff_lb.reshape(np.asarray(v.lb).shape),
                eff_ub.reshape(np.asarray(v.ub).shape),
            )
    return box


__all__ = ["ConvexReformPass"]
