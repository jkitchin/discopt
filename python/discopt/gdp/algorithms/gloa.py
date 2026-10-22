"""Global logic-based outer approximation (#1775 G4.3, spec §5.3; Lee & Grossmann 2001).

:class:`GLOA` is :class:`~discopt.gdp.algorithms.loa.LOA` with two changes:

* **The master is a relaxation of every row.** It is the MILP relaxation of the
  lowered model (:func:`~discopt._relax.milp_relaxation.build_milp_relaxation`:
  McCormick/αBB/secant envelopes of each nonlinear term over the global box),
  wrapped as a :class:`~discopt.gdp.algorithms.master.RelaxationMaster`. A
  nonconvex row inside a disjunct is relaxed and then big-M'd onto that
  disjunct's indicator by the lowering, which is the master of the spec. LOA's
  OA cuts are kept: they go only on certified-convex rows and are valid there.
* **Subproblems are global.** A configuration is excluded from the master only
  on a certificate (optimal or infeasible); anything less is unresolved and
  blocks ``optimal`` (the RIC no-good rule).

Bound (§5.4): every master row is a valid relaxation and every no-good removes a
configuration that is either certified infeasible or whose certified bound is
kept in ``excluded_floor``, so ``min(master bound, excluded_floor)`` is a valid
LB with no convexity certificate. Pyomo's GLOA builds its convex-side cut from
the concave subgradient (spec §3.1); discopt's relaxation layer emits each side
from its own envelope.

Not yet (spec §5.3, later work): the disjunct-local box of G5.1 (the relaxation
is over the global box) and relaxation cuts re-linearized at the subproblem
points (the relaxation is built once). Infinite bounds are not refused, unlike
the spec's §0.2.5 wording: discopt's relaxation never invents a bound (spec §3.3
is Pyomo's MC++ doing so); a term over an unbounded box gets a weaker or no
envelope, which loosens the bound and never invalidates it.
"""

from __future__ import annotations

from typing import Any

from discopt.gdp.algorithms.loa import LOA
from discopt.gdp.algorithms.master import MasterMILP, RelaxationMaster
from discopt.modeling.core import Model


class GLOA(LOA):
    """Global LOA; see the module docstring."""

    route = (
        "gdpopt-gloa: global logic-based outer approximation "
        "(relaxation MILP master + global subproblems)"
    )

    def __init__(self, model: Model, *, subproblem: str = "global", **kw: Any) -> None:
        if subproblem != "global":
            raise ValueError(
                "GLOA's subproblems are global: a configuration leaves the master only "
                "on a certificate (subproblem must be 'global')"
            )
        self.relaxation_truncated = False
        self.relaxation_aborted = False
        super().__init__(model, subproblem="global", **kw)

    def _make_master(self, milp_solver: str) -> MasterMILP:
        from discopt._relax.discretization import DiscretizationState
        from discopt._relax.milp_relaxation import build_milp_relaxation
        from discopt._relax.term_classifier import classify_nonlinear_terms
        from discopt._relax.uniform_relax import RelaxationBuildAborted

        low = self.space.lowered
        linear = super()._make_master(milp_solver)
        self.model_objective_linear = linear.objective is not None
        # The build gets the caller's whole deadline (#912 ``contract``): one
        # stopped early between rows is a weaker relaxation, still a valid one;
        # one aborted (the objective walk itself overran) is no relaxation, and
        # GLOA keeps LOA's master -- its linear rows, valid on their own.
        at = self.deadline.at()
        try:
            relaxation, _ = build_milp_relaxation(
                low,
                classify_nonlinear_terms(low),
                DiscretizationState(),
                build_deadline=at,
                abort_deadline=at,
            )
        except RelaxationBuildAborted:
            self.relaxation_aborted = True
            return linear
        master = RelaxationMaster(
            relaxation,
            self.n,
            objective_fallback=linear.objective,
            backend=milp_solver,
        )
        self.relaxation_truncated = master.truncated
        if master.truncated:
            # A truncated build may stop before the selection rows; the lowered
            # model's own linear rows are exact, so they go back in.
            for row, sense, rhs in zip(linear.rows, linear.senses, linear.rhs):
                master.add_row(master.model_row(row), sense, rhs)
        return master

    def _cuts_at(self, x: Any) -> None:
        # LOA's cuts are over the model's columns (``(x, eta)`` for an
        # objective epigraph cut); ``add_model_cut`` writes them over the
        # relaxation's, and ``add_cut`` validates each one.
        from discopt.solvers.gdpopt_loa import _add_oa_cuts

        rows: list = []
        rhs: list = []
        _add_oa_cuts(
            self.evaluator,
            x,
            self.n,
            int(self.evaluator.n_constraints),
            rows,
            rhs,
            self.obj_is_linear,
            self.oa_rows,
            self.master.objective_convex,
        )
        master = self.master
        for row, b in zip(rows, rhs):
            if isinstance(master, RelaxationMaster):
                master.add_model_cut(row, b, epigraph=len(row) == self.n + 1)
            else:  # the build was aborted: LOA's own master, LOA's own cuts
                master.add_cut(row, b)
        self._objective_cut_at(x)

    def _objective_cut_at(self, x: Any) -> None:
        """LOA's objective tangent, written on the relaxation's objective.

        The master minimizes ``c @ w + offset``, the relaxation's stand-in for the
        model's objective ``f`` (minimization convention). For every feasible
        ``x`` some relaxation point has ``c @ w + offset = f(x)``, and a
        certified-convex ``f`` lies above its tangent at ``x_bar``, so
        ``c @ w + offset >= f(x_bar) + grad f(x_bar) (x - x_bar)`` cuts nothing
        feasible. Without it the master bounds a convex objective only by its
        envelope over the box, looser than LOA's tangents (measured: positioning
        and small_batch, which LOA certifies).
        """
        import numpy as np

        from discopt._relax.cutting_planes import generate_objective_oa_cut

        master = self.master
        if (
            self.model_objective_linear
            or not self.objective_certified_convex
            or not getattr(master, "objective_from_relaxation", False)
            or master.objective is None
        ):
            return
        cut = generate_objective_oa_cut(self.evaluator, x, self.n + 1, z_index=self.n)
        if cut is None:
            return
        c, offset = master.objective
        coef_z = float(cut.coeffs[self.n])  # -1: grad . x - z <= grad . x_bar - f(x_bar)
        row = coef_z * np.asarray(c, dtype=np.float64)
        row[: self.n] += cut.coeffs[: self.n]
        master.add_cut(row, float(cut.rhs) - coef_z * float(offset))

    def extra_stats(self) -> dict[str, Any]:
        return {
            "relaxation_columns": int(self.master.n_vars),
            "relaxation_truncated": bool(self.relaxation_truncated),
            "relaxation_aborted": bool(self.relaxation_aborted),
            "cleaned_cut_coefficients": int(getattr(self.master, "cleaned_coefficients", 0)),
        }


def solve_gloa(model: Model, **options: Any):
    """Run :class:`GLOA` on ``model`` and return its ``SolveResult``."""
    return GLOA(model, **options).run()


__all__ = ["GLOA", "solve_gloa"]
