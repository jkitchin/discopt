"""L-shaped method (probability-weighted multicut Benders) for two-stage SP.

The L-shaped method *is* Benders decomposition applied to the two-stage stochastic
program: the first stage is the master, each scenario's recourse is a subproblem,
and optimality/feasibility cuts approximate the expected recourse
``Q(x) = Σ_s p_s Q_s(x)``.

**Probability weighting without an engine change.** The plan's one "missing
primitive" is that the decomposition engine sums block contributions with weight
1.0. We satisfy it entirely *in the model*: the extensive form built with
:class:`~discopt.stochastic.risk.Expectation` already scales each scenario's
recourse cost by ``p_s`` in the objective. Benders reads those objective
coefficients, so subproblem ``s`` optimizes ``p_s Q_s`` and its cut is
``p_s``-weighted — the master's ``Σ_s η_s`` converges to ``Σ_s p_s Q_s`` with the
engine unchanged. So the L-shaped driver is: build the weighted extensive form,
annotate the first-stage (complicating) variables, and call ``solve_benders``.

Phase 1 is **risk-neutral** (Expectation). Risk-averse L-shaped (CVaR couples
scenarios through ``η``) is later. See ``docs/dev/stochastic-module-plan.md`` §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from discopt.modeling.core import Expression, Model, Variable
from discopt.stochastic.extensive_form import ExtensiveForm, build_extensive_form
from discopt.stochastic.risk import Expectation
from discopt.stochastic.scenario import ScenarioSet

__all__ = ["LShapedResult", "solve_lshaped"]


@dataclass
class LShapedResult:
    """Result of :func:`solve_lshaped`."""

    extensive_form: ExtensiveForm
    structure: object  # DecompositionStructure (scenarios = recourse blocks)
    first_stage_vars: list
    result: object | None  # the solve_benders SolveResult, or None if solve=False


def solve_lshaped(
    model: Model,
    *,
    first_stage_vars: list[Variable],
    scenarios: ScenarioSet,
    recourse_builder,
    first_stage_cost: Expression | None = None,
    risk=None,
    method: str = "auto",
    solve: bool = True,
    **benders_kwargs,
) -> LShapedResult:
    """Set up and (optionally) solve a two-stage SP by the L-shaped method.

    Parameters mirror :func:`~discopt.stochastic.extensive_form.build_extensive_form`
    plus ``first_stage_vars`` (the here-and-now / complicating variables). Extra
    keyword arguments are forwarded to
    :func:`discopt.decomposition.solve_benders`.

    ``method`` selects the per-node engine: ``"benders"`` (classical L-shaped, LP
    recourse), ``"gbd"`` (Generalized Benders — **convex-nonlinear recourse**), or
    ``"auto"`` (``solve_benders`` auto-dispatches to GBD when the model is nonlinear).

    Set ``solve=False`` to build the weighted extensive form and its decomposition
    structure without invoking the (backend-dependent) Benders solve — useful for
    inspecting or asserting the decomposition.
    """
    if method not in ("auto", "benders", "gbd"):
        raise ValueError(f"method must be 'auto', 'benders', or 'gbd', got {method!r}")
    risk = risk or Expectation()
    if not isinstance(risk, Expectation):
        raise NotImplementedError(
            "Phase 1 L-shaped is risk-neutral (Expectation). CVaR / chance-constrained "
            "L-shaped couples scenarios through the risk auxiliaries and is a later "
            "phase; use build_extensive_form for a risk-averse deterministic equivalent."
        )

    ef = build_extensive_form(
        model,
        scenarios=scenarios,
        recourse_builder=recourse_builder,
        first_stage_cost=first_stage_cost,
        risk=risk,
    )

    # Annotate the first stage; the scenario recourse blocks separate automatically.
    model.first_stage(*first_stage_vars)

    from discopt.decomposition import detect_decomposition

    fs_names = [v.name for v in first_stage_vars]
    structure = detect_decomposition(model, complicating=fs_names)

    result = None
    if solve:
        if method == "gbd":
            from discopt.decomposition import solve_gbd

            result = solve_gbd(model, structure=structure, **benders_kwargs)
        else:  # "auto" / "benders" — solve_benders auto-dispatches to GBD if nonlinear
            from discopt.decomposition import solve_benders

            result = solve_benders(model, structure=structure, **benders_kwargs)

    return LShapedResult(
        extensive_form=ef,
        structure=structure,
        first_stage_vars=list(first_stage_vars),
        result=result,
    )
