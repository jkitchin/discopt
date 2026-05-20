"""``discopt doe`` — Excel-workbook-driven optimal experimental design.

Five verbs make up the loop:

* ``discopt doe templates`` — list the available template models.
* ``discopt doe new TEMPLATE [args] -o file.xlsx --n N`` — start a
  campaign with N optimal initial runs.
* ``discopt doe status file.xlsx`` — show how many runs are complete,
  pending, and what to do next.
* ``discopt doe fit file.xlsx`` — estimate parameters from completed
  runs and refresh the FIM.
* ``discopt doe extend file.xlsx --n M`` — append M more optimal runs
  using the cumulative FIM as the prior.

Each verb is split into a pure ``do_<verb>(params: dict) -> dict``
function and a ``_cmd_<verb>(args)`` argparse wrapper. The pure
functions raise typed exceptions on failure and are the contract a
future GUI binds to. The wrappers add argparse parsing, ``--json``
output formatting, and exit codes — nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from discopt.doe.templates import (
    TEMPLATE_NAMES,
    build_template,
    template_parameter_names,
)
from discopt.doe.workbook import InputSpec, Workbook, _load_module_callable

# Tiny ridge added to the prior FIM whenever no fitted prior exists.
# Keeps log-det-FIM finite for the very first batch of a linear-in-
# parameters template (rank-1 single-design FIM is otherwise singular).
_RIDGE = 1e-6

_DEFAULT_CRITERION = "determinant"
_CRITERION_CHOICES = ("determinant", "trace", "min_eigenvalue", "condition_number")
_CRITERION_ALIASES = {
    "D": "determinant",
    "A": "trace",
    "E": "min_eigenvalue",
    "ME": "condition_number",
}


class DoEError(Exception):
    """Raised by ``do_*`` functions on user-facing failures."""


# ──────────────────────────────────────────────────────────────────
# argparse parsing helpers
# ──────────────────────────────────────────────────────────────────


def _parse_input_spec(s: str) -> tuple[str, float, float]:
    """Parse ``name:lb:ub`` (used by ``--input`` / ``--bounds``)."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'name:lb:ub', got {s!r}")
    name, lb_s, ub_s = parts
    if not name:
        raise argparse.ArgumentTypeError(f"input name is empty in {s!r}")
    try:
        lb = float(lb_s)
        ub = float(ub_s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad bound in {s!r}: {e}") from e
    if not (ub > lb):
        raise argparse.ArgumentTypeError(f"input {name!r}: upper bound must exceed lower bound")
    return (name, lb, ub)


def _parse_levels_spec(s: str) -> tuple[str, list[object]]:
    """Parse ``name:L1,L2,L3`` for Latin-family --levels arguments.

    Levels are interpreted as floats when all parts parse as numeric,
    otherwise as strings.
    """
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"expected 'name:L1,L2,...', got {s!r}")
    name, _, levels_s = s.partition(":")
    if not name:
        raise argparse.ArgumentTypeError(f"factor name is empty in {s!r}")
    raw = [p.strip() for p in levels_s.split(",") if p.strip()]
    if len(raw) < 2:
        raise argparse.ArgumentTypeError(f"factor {name!r}: need at least 2 levels")
    try:
        levels: list[object] = [float(p) for p in raw]
    except ValueError:
        levels = list(raw)
    return (name, levels)


def _parse_factor_pair(s: str) -> tuple[str, object, object]:
    """Parse ``name:LOW:HIGH`` where LOW/HIGH may be numeric or string."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'name:LOW:HIGH', got {s!r}")
    name, lo_s, hi_s = parts
    if not name:
        raise argparse.ArgumentTypeError(f"factor name is empty in {s!r}")

    def _maybe_num(v: str) -> object:
        try:
            return float(v)
        except ValueError:
            return v

    lo, hi = _maybe_num(lo_s), _maybe_num(hi_s)
    if lo == hi:
        raise argparse.ArgumentTypeError(f"factor {name!r}: LOW and HIGH must differ")
    return (name, lo, hi)


def _parse_kv_float(s: str) -> tuple[str, float]:
    """Parse ``name=value`` (used by ``--params``)."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected 'name=value', got {s!r}")
    name, _, val = s.partition("=")
    try:
        return (name, float(val))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad numeric value in {s!r}: {e}") from e


def _normalize_criterion(c: str) -> str:
    return _CRITERION_ALIASES.get(c, c)


# ──────────────────────────────────────────────────────────────────
# Typed parameter / result records
# ──────────────────────────────────────────────────────────────────


@dataclass
class NewParams:
    output: Path
    n: int
    inputs: list[tuple[str, float, float]]
    response_name: str
    measurement_error: float
    criterion: str
    seed: int
    n_starts: int
    template: str | None = None
    degree: int | None = None
    mixture_total: float | None = None
    module_callable: str | None = None
    param_initial_guess: dict[str, float] = field(default_factory=dict)
    levels: dict[str, list[object]] | None = None
    replicates: int = 1
    factor_pairs: dict[str, tuple[object, object]] | None = None
    center_points: int = 0
    # Active-learning ("optimize") template options
    optimize_criterion: str = "maximize"
    optimize_surrogate: str = "gp"
    optimize_acquisition: str = "expected_improvement"


@dataclass
class OptimizeParams:
    """Inputs to :func:`do_optimize`."""

    workbook: Path
    criterion: str = "maximize"
    surrogate: str = "gp"
    acquisition: str = "expected_improvement"
    batch_size: int = 4
    n_candidates: int = 2048
    seed: int | None = None
    acquisition_kwargs: dict[str, float] | None = None
    custom_surrogate_path: str | None = None
    custom_surrogate_kwargs: dict[str, Any] | None = None


@dataclass
class ExtendParams:
    workbook: Path
    n: int
    n_starts: int


# ──────────────────────────────────────────────────────────────────
# Internal: build Experiment + parameter-name list from spec
# ──────────────────────────────────────────────────────────────────


def _build_experiment_from_new(params: NewParams):
    """Return ``(experiment, parameter_names)`` for a ``do_new`` request."""
    if params.template is not None and params.module_callable is not None:
        raise DoEError("specify either --template or --module, not both")
    if params.template is None and params.module_callable is None:
        raise DoEError("specify a template name or --module")

    if params.template is not None:
        if params.template not in TEMPLATE_NAMES:
            raise DoEError(
                f"unknown template {params.template!r}; choose from {list(TEMPLATE_NAMES)}"
            )
        exp = build_template(
            params.template,
            inputs=params.inputs,
            response_name=params.response_name,
            measurement_error=params.measurement_error,
            degree=params.degree,
            mixture_total=params.mixture_total,
        )
        names = template_parameter_names(
            params.template, degree=params.degree, n_inputs=len(params.inputs)
        )
        return exp, names

    exp = _load_module_callable(params.module_callable)  # type: ignore[arg-type]
    em = exp.create_model(**params.param_initial_guess)
    return exp, em.parameter_names


def _mixture_constraints(
    template: str | None,
    inputs: list[tuple[str, float, float]],
    mixture_total: float | None,
):
    """Return ``(equality_constraints, feasible_projection)`` for Scheffé
    templates, or ``(None, None)`` for any other template.

    Mixture designs require an equality constraint ``sum(components) ==
    total`` plus a Dirichlet-style projection so the multi-start lands
    on the simplex from the start.
    """
    if not template or not template.startswith("scheffe-"):
        return None, None
    from discopt.doe import project_to_simplex, sum_constraint

    components = [name for name, _, _ in inputs]
    bounds = {name: (lb, ub) for name, lb, ub in inputs}
    total = float(mixture_total) if mixture_total is not None else 1.0
    g = sum_constraint(components, total=total)

    def proj(point: dict[str, float]) -> dict[str, float]:
        return project_to_simplex(point, components, total=total, bounds=bounds)

    return [g], proj


def _design_bounds(specs: list[tuple[str, float, float]]) -> dict[str, tuple[float, float]]:
    return {name: (lb, ub) for name, lb, ub in specs}


def _param_values_for_design(
    parameter_names: list[str],
    workbook: Workbook | None,
) -> dict[str, float]:
    """Pick the parameter point at which to evaluate FIMs.

    Order of precedence: fitted parameters from the workbook
    (if available), then ``param_initial_guess`` metadata, then zeros.
    """
    if workbook is not None:
        fitted = workbook.read_parameters()
        if fitted:
            out: dict[str, float] = {}
            for row in fitted:
                if row["estimate"] is not None:
                    out[row["name"]] = float(row["estimate"])
            if out:
                for name in parameter_names:
                    out.setdefault(name, 0.0)
                return out
        guesses = workbook.param_initial_guess()
        if guesses:
            out = {name: float(guesses.get(name, 0.0)) for name in parameter_names}
            return out
    return {name: 0.0 for name in parameter_names}


def _cumulative_fim_from_completed(
    experiment: Any,
    parameter_names: list[str],
    completed_runs: list[dict[str, Any]],
    input_names: list[str],
) -> np.ndarray:
    """Sum FIMs over every completed run. Adds a ridge so first-batch
    FIMs stay non-singular when fewer runs than parameters have come in.
    """
    from discopt.doe.fim import compute_fim

    n_p = len(parameter_names)
    fim = _RIDGE * np.eye(n_p)
    param_values = {name: 0.0 for name in parameter_names}
    for row in completed_runs:
        design = {nm: float(row[nm]) for nm in input_names}
        try:
            r = compute_fim(experiment, param_values, design)
        except Exception:
            continue
        fim = fim + np.asarray(r.fim)
    return fim


# ──────────────────────────────────────────────────────────────────
# do_templates
# ──────────────────────────────────────────────────────────────────


_TEMPLATE_DESCRIPTIONS = {
    "linear": (
        "y = b0 + sum_i bi * xi.  Inputs: --input NAME:LB:UB (repeatable). "
        "Parameters: 1 + n_inputs."
    ),
    "polynomial-1d": (
        "y = sum_{j=0..d} bj * x**j.  Inputs: --input NAME:LB:UB (single), "
        "--degree D.  Parameters: D + 1."
    ),
    "response-surface-2d": (
        "Full quadratic in 2 factors (intercept, 2 main, 2 square, 1 cross). "
        "Inputs: --input NAME:LB:UB (exactly 2).  Parameters: 6."
    ),
    "response-surface-3d": (
        "Full quadratic in 3 factors (intercept, 3 main, 3 square, 3 cross). "
        "Inputs: --input NAME:LB:UB (exactly 3).  Parameters: 10."
    ),
    "scheffe-linear": (
        "Scheffé canonical linear mixture: y = sum_i bi * xi, with "
        "sum(components) == --mixture-total.  Inputs: --input NAME:LB:UB "
        "(>= 2 components), --mixture-total T (default 1.0).  Parameters: q."
    ),
    "scheffe-quadratic": (
        "Scheffé canonical quadratic mixture: pure-blend + pairwise "
        "blending terms.  Inputs: --input NAME:LB:UB (>= 2), "
        "--mixture-total T.  Parameters: q + q(q-1)/2."
    ),
    "scheffe-special-cubic": (
        "Scheffé canonical special-cubic mixture: adds three-way blending "
        "to the quadratic.  Inputs: --input NAME:LB:UB (>= 3), "
        "--mixture-total T.  Parameters: q + q(q-1)/2 + q(q-1)(q-2)/6."
    ),
    "latin-square": (
        "Latin-square design for ANOVA on 3 factors (1 treatment + 2 blocks). "
        "Inputs: --levels NAME:L1,L2,... (exactly 3 factors, same level count). "
        "Optional: --replicates R."
    ),
    "graeco-latin": (
        "Graeco-Latin square for ANOVA on 4 factors. Inputs: --levels "
        "NAME:L1,L2,... (exactly 4 factors, k>=3 and k!=6). "
        "Optional: --replicates R."
    ),
    "hyper-graeco-latin": (
        "Hyper-Graeco-Latin square for ANOVA on 5 factors (3 MOLS). Inputs: "
        "--levels NAME:L1,L2,... (exactly 5 factors, k in {4,5,7,...}). "
        "Optional: --replicates R."
    ),
    "factorial-2level": (
        "2-level full factorial for screening. Inputs: --factor "
        "NAME:LOW:HIGH (repeatable, 2-8 factors; LOW/HIGH may be "
        "numeric or string). Optional: --center-points N (numeric "
        "factors only), --replicates R. Use this to answer 'does each "
        "factor matter?' before fitting a response surface."
    ),
    "optimize": (
        "Active-learning workbook for sequential optimization. Inputs: "
        "--input NAME:LB:UB (repeatable). The seed batch is sampled by "
        "Sobol; subsequent rounds are produced by 'discopt doe optimize "
        "WORKBOOK' which fits a surrogate and proposes new runs via the "
        "chosen acquisition. Optional: --optimize-criterion "
        "{maximize,minimize}, --optimize-surrogate (preset name), "
        "--optimize-acquisition."
    ),
}


def do_templates(_params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "templates": [
            {"name": name, "description": _TEMPLATE_DESCRIPTIONS[name]} for name in TEMPLATE_NAMES
        ]
    }


def _cmd_templates(args) -> int:
    out = do_templates()
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for entry in out["templates"]:
            print(f"{entry['name']}")
            print(f"    {entry['description']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# do_new
# ──────────────────────────────────────────────────────────────────


def do_new(params: NewParams) -> dict[str, Any]:
    from discopt.doe import batch_optimal_experiment
    from discopt.doe.templates import COMBINATORIAL_TEMPLATES

    if params.template == "factorial-2level":
        return _do_new_factorial(params)
    if params.template == "optimize":
        return _do_new_optimize(params)
    if params.template in COMBINATORIAL_TEMPLATES:
        return _do_new_latin(params)

    experiment, parameter_names = _build_experiment_from_new(params)
    bounds = _design_bounds(params.inputs)
    input_names = [s[0] for s in params.inputs]

    n_p = len(parameter_names)
    prior_fim = _RIDGE * np.eye(n_p)

    # For module-callable (likely nonlinear) experiments, use any
    # supplied parameter guesses; for templates, zeros are fine (FIM
    # independent of params in linear-in-parameters models).
    param_values = (
        {name: float(params.param_initial_guess.get(name, 0.0)) for name in parameter_names}
        if params.module_callable
        else {name: 0.0 for name in parameter_names}
    )

    eq_cons, proj = _mixture_constraints(params.template, params.inputs, params.mixture_total)

    if params.n == 1:
        from discopt.doe import optimal_experiment

        single = optimal_experiment(
            experiment,
            param_values,
            bounds,
            criterion=params.criterion,
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=params.seed,
        )
        designs = [single.design]
        criterion_value = float(single.criterion_value)
    else:
        batch = batch_optimal_experiment(
            experiment,
            param_values,
            bounds,
            n_experiments=params.n,
            criterion=params.criterion,
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=params.seed,
        )
        designs = list(batch.designs)
        criterion_value = float(batch.criterion_value)

    template_args: dict[str, Any] = {}
    if params.degree is not None:
        template_args["degree"] = int(params.degree)
    if params.mixture_total is not None:
        template_args["mixture_total"] = float(params.mixture_total)

    wb = Workbook.create(
        params.output,
        template=params.template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion=params.criterion,
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=params.module_callable,
        param_initial_guess=params.param_initial_guess or None,
    )
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": params.template,
            "module_callable": params.module_callable,
            "n": params.n,
            "criterion": params.criterion,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": params.template,
        "module_callable": params.module_callable,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": params.criterion,
        "criterion_value": criterion_value,
        "parameter_names": parameter_names,
        "n_parameters": len(parameter_names),
        "next_command": f"discopt doe status {wb.path}",
    }


def _do_new_factorial(params: NewParams) -> dict[str, Any]:
    """Build a 2-level full factorial design and persist as a workbook."""
    from discopt.doe import factorial_2level_design

    pairs = params.factor_pairs
    if not pairs:
        raise DoEError("factorial-2level requires --factor NAME:LOW:HIGH (repeatable)")
    if len(pairs) < 2:
        raise DoEError("factorial-2level needs at least 2 factors")
    if len(pairs) > 8:
        raise DoEError("factorial-2level supports up to 8 factors")

    design = factorial_2level_design(
        pairs,
        center_points=params.center_points,
        replicates=params.replicates,
        seed=params.seed,
    )

    factor_names = list(pairs.keys())
    input_specs_data: list[tuple[str, float, float]] = []
    for name in factor_names:
        lo, hi = pairs[name]
        if (
            isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
            and not isinstance(lo, bool)
        ):
            lb, ub = float(min(lo, hi)), float(max(lo, hi))
            if ub == lb:
                ub = lb + 1.0
        else:
            lb, ub = 0.0, 1.0
        input_specs_data.append((name, lb, ub))

    template_args: dict[str, Any] = {
        "levels": {name: [pairs[name][0], pairs[name][1]] for name in factor_names},
        "replicates": int(params.replicates),
        "center_points": int(params.center_points),
        "family": "factorial-2level",
    }

    wb = Workbook.create(
        params.output,
        template=params.template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in input_specs_data],
        criterion="anova",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
    )

    designs = [{n: row[n] for n in factor_names} for row in design.rows]
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": "factorial-2level",
            "n": len(designs),
            "replicates": params.replicates,
            "center_points": params.center_points,
            "k": len(factor_names),
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": "factorial-2level",
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{n: design.rows[i][n] for n in factor_names}}
            for i, rid in enumerate(new_ids)
        ],
        "criterion": "anova",
        "criterion_value": float("nan"),
        "parameter_names": [],
        "n_parameters": 0,
        "next_command": f"discopt doe anova {wb.path}",
    }


def _do_new_optimize(params: NewParams) -> dict[str, Any]:
    """Create a workbook seeded with an initial Sobol batch for active learning.

    The workbook stores the box bounds (``input_specs``) so subsequent
    rounds of :func:`do_optimize` can sample candidates inside them.
    The seed batch is generated by Sobol sampling and gives the
    surrogate something to fit against.
    """
    if not params.inputs:
        raise DoEError("optimize template requires --input NAME:LB:UB (at least one)")
    if int(params.n) < 2:
        raise DoEError("optimize seed batch must have n >= 2 runs")

    bounds = np.array([(lb, ub) for _, lb, ub in params.inputs], dtype=float)
    input_names = [n_ for n_, _, _ in params.inputs]

    try:
        from scipy.stats import qmc

        engine = qmc.Sobol(d=len(input_names), scramble=True, seed=int(params.seed))
        m = max(1, int(math.ceil(math.log2(int(params.n)))))
        unit = engine.random_base2(m=m)[: int(params.n)]
    except ImportError:
        rng = np.random.default_rng(int(params.seed))
        unit = rng.uniform(size=(int(params.n), len(input_names)))

    lo, hi = bounds[:, 0], bounds[:, 1]
    seed_points = lo + (hi - lo) * unit
    designs = [
        {nm: float(seed_points[i, j]) for j, nm in enumerate(input_names)}
        for i in range(int(params.n))
    ]

    template_args: dict[str, Any] = {
        "family": "optimize",
        "criterion": params.optimize_criterion,
        "surrogate": params.optimize_surrogate,
        "acquisition": params.optimize_acquisition,
    }

    wb = Workbook.create(
        params.output,
        template="optimize",
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in params.inputs],
        criterion="active-learning",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
    )
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": "optimize",
            "n": int(params.n),
            "criterion": params.optimize_criterion,
            "surrogate": params.optimize_surrogate,
            "acquisition": params.optimize_acquisition,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": "optimize",
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [{"run_id": rid, **designs[i]} for i, rid in enumerate(new_ids)],
        "criterion": params.optimize_criterion,
        "criterion_value": float("nan"),
        "parameter_names": [],
        "n_parameters": 0,
        "next_command": f"discopt doe optimize {wb.path}",
    }


def do_optimize(params: OptimizeParams) -> dict[str, Any]:
    """Run one active-learning round on an optimize-template workbook.

    Reads completed runs from ``params.workbook``, fits the chosen
    surrogate, scores Sobol candidates with the chosen acquisition,
    and appends the recommended batch as pending runs.
    """
    from discopt.doe import (
        OptimizationCriterion,
    )
    from discopt.doe import (
        optimize_round as _optimize_round,
    )

    wb = Workbook.open(Path(params.workbook))
    if (wb.template_name() or "") != "optimize":
        raise DoEError(
            f"workbook template is {wb.template_name()!r}, not 'optimize'; "
            "use `discopt doe new optimize` to create one"
        )

    surrogate_obj: object = params.surrogate
    if params.custom_surrogate_path:
        surrogate_obj = _instantiate_dotted(
            params.custom_surrogate_path, params.custom_surrogate_kwargs or {}
        )

    result = _optimize_round(
        workbook=wb,
        criterion=OptimizationCriterion(params.criterion),
        surrogate=surrogate_obj,
        acquisition=params.acquisition,
        batch_size=int(params.batch_size),
        n_candidates=int(params.n_candidates),
        seed=params.seed,
        acquisition_kwargs=params.acquisition_kwargs,
    )

    return {
        "workbook_path": str(wb.path),
        "criterion": params.criterion,
        "acquisition": params.acquisition,
        "surrogate": (
            params.custom_surrogate_path if params.custom_surrogate_path else params.surrogate
        ),
        "surrogate_mode": result.surrogate_mode,
        "batch_size": int(params.batch_size),
        "n_completed": result.n_completed,
        "incumbent_x": result.incumbent_x,
        "incumbent_y": result.incumbent_y,
        "new_run_ids": result.new_run_ids,
        "next_designs": [
            {"run_id": rid, **d} for rid, d in zip(result.new_run_ids, result.next_designs)
        ],
        "acquisition_scores": result.acquisition_scores,
    }


def _instantiate_dotted(spec: str, kwargs: dict[str, Any]) -> object:
    """Import and instantiate ``module.path:ClassName`` or ``module.path.ClassName``.

    ``spec`` may use either ``:`` (preferred) or the last ``.`` as the
    separator between module path and class name. ``kwargs`` are
    forwarded to the constructor.
    """
    import importlib

    if ":" in spec:
        mod_path, cls_name = spec.split(":", 1)
    else:
        if "." not in spec:
            raise DoEError(
                f"custom surrogate spec {spec!r} must be 'module.path:ClassName' "
                "or 'module.path.ClassName'"
            )
        mod_path, _, cls_name = spec.rpartition(".")
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        raise DoEError(f"could not import {mod_path!r}: {e}") from e
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise DoEError(f"{cls_name!r} not found in {mod_path!r}") from e
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise DoEError(f"could not instantiate {spec} with kwargs {kwargs!r}: {e}") from e


_LATIN_FACTOR_COUNT = {
    "latin-square": 3,
    "graeco-latin": 4,
    "hyper-graeco-latin": 5,
}


def _do_new_latin(params: NewParams) -> dict[str, Any]:
    """Build a Latin-family design (closed-form) and persist as a workbook.

    Latin family designs do not have a FIM-based optimization step;
    runs come from the combinatorial generator and the workbook is
    used only for capturing responses + downstream ANOVA.
    """
    from discopt.doe import latin_square_design

    if not params.template or params.template not in _LATIN_FACTOR_COUNT:
        raise DoEError(f"unsupported latin template {params.template!r}")
    if not params.levels:
        raise DoEError(f"{params.template} requires --levels NAME:L1,L2,... per factor")

    expected = _LATIN_FACTOR_COUNT[params.template]
    if len(params.levels) != expected:
        raise DoEError(
            f"{params.template} requires exactly {expected} factors, got {len(params.levels)}"
        )

    level_counts = {len(v) for v in params.levels.values()}
    if len(level_counts) != 1:
        raise DoEError("all factors must have the same number of levels")
    k = level_counts.pop()
    if k < 2:
        raise DoEError("each factor needs at least 2 levels")
    if params.template != "latin-square" and k == 6:
        raise DoEError(
            f"{params.template}: no orthogonal Latin squares exist for k = 6 (Euler exception)"
        )

    design = latin_square_design(params.levels, replicates=params.replicates, seed=params.seed)

    # Synthesize numeric (lb, ub) input specs for compatibility with the
    # rest of the workbook code. For numeric levels we use min/max; for
    # categorical levels we use 0..k-1 placeholders.
    factor_names = list(params.levels.keys())
    input_specs_data: list[tuple[str, float, float]] = []
    for name in factor_names:
        vals = params.levels[name]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            numeric_vals = [float(cast(float, v)) for v in vals]
            lb, ub = min(numeric_vals), max(numeric_vals)
            if ub == lb:
                ub = lb + 1.0
        else:
            lb, ub = 0.0, float(len(vals) - 1)
        input_specs_data.append((name, lb, ub))

    template_args: dict[str, Any] = {
        "levels": {name: list(params.levels[name]) for name in factor_names},
        "replicates": int(params.replicates),
        "family": design.family,
    }

    wb = Workbook.create(
        params.output,
        template=params.template,
        template_args=template_args,
        input_specs=[InputSpec(n_, lb, ub) for n_, lb, ub in input_specs_data],
        criterion="anova",
        measurement_error=params.measurement_error,
        seed=params.seed,
        response_name=params.response_name,
        module_callable=None,
        param_initial_guess=None,
    )

    designs = [{n: row[n] for n in factor_names} for row in design.rows]
    new_ids = wb.append_runs(1, designs)
    wb.log(
        "new",
        {
            "template": params.template,
            "n": len(designs),
            "replicates": params.replicates,
            "k": k,
            "family": design.family,
        },
    )
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "template": params.template,
        "module_callable": None,
        "batch": 1,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{n: design.rows[i][n] for n in factor_names}}
            for i, rid in enumerate(new_ids)
        ],
        "criterion": "anova",
        "criterion_value": float("nan"),
        "parameter_names": [],
        "n_parameters": 0,
        "next_command": f"discopt doe anova {wb.path}",
    }


def _cmd_new(args) -> int:
    output = Path(args.output)
    if output.exists() and not args.force:
        return _fail(
            args,
            f"{output} already exists. Pass --force to overwrite.",
            workbook_path=str(output),
        )
    is_module = bool(getattr(args, "_is_module", False))
    inputs: list[tuple[str, float, float]] = list(getattr(args, "input", None) or []) + list(
        getattr(args, "bounds", None) or []
    )
    levels_arg = getattr(args, "levels", None)
    levels_dict: dict[str, list[object]] | None = None
    if levels_arg:
        levels_dict = {name: list(vals) for name, vals in levels_arg}
    factor_arg = getattr(args, "factor", None)
    factor_pairs: dict[str, tuple[object, object]] | None = None
    if factor_arg:
        factor_pairs = {name: (lo, hi) for name, lo, hi in factor_arg}
    params = NewParams(
        output=output,
        n=int(args.n),
        inputs=inputs,
        response_name=args.response,
        measurement_error=float(args.error),
        criterion=_normalize_criterion(args.criterion),
        seed=int(args.seed),
        n_starts=int(args.n_starts),
        template=None if is_module else getattr(args, "template", None),
        degree=getattr(args, "degree", None),
        mixture_total=getattr(args, "mixture_total", None),
        module_callable=getattr(args, "module", None) if is_module else None,
        param_initial_guess=dict(args.params or []) if is_module else {},
        levels=levels_dict,
        replicates=int(getattr(args, "replicates", 1) or 1),
        factor_pairs=factor_pairs,
        center_points=int(getattr(args, "center_points", 0) or 0),
        optimize_criterion=getattr(args, "optimize_criterion", "maximize") or "maximize",
        optimize_surrogate=getattr(args, "optimize_surrogate", "gp") or "gp",
        optimize_acquisition=(
            getattr(args, "optimize_acquisition", "expected_improvement") or "expected_improvement"
        ),
    )
    try:
        out = do_new(params)
    except (DoEError, ValueError, TypeError, FileNotFoundError, ImportError) as e:
        return _fail(args, str(e), workbook_path=str(output))
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print_new_human(out)
    return 0


def _print_new_human(out: dict[str, Any]) -> None:
    print(f"created workbook: {out['workbook_path']}")
    label = out["template"] or out["module_callable"]
    print(f"  model:         {label}")
    print(f"  parameters:    {out['n_parameters']} ({', '.join(out['parameter_names'])})")
    print(f"  criterion:     {out['criterion']}")
    new_ids = out["new_run_ids"]
    print(f"  batch:         {out['batch']}  (run_ids {min(new_ids)}-{max(new_ids)})")
    print("  recommended runs:")
    if out["designs"]:
        keys = [k for k in out["designs"][0].keys() if k != "run_id"]
        header = "    run_id  " + "  ".join(f"{k:>10s}" for k in keys)
        print(header)
        for row in out["designs"]:
            cells = []
            for k in keys:
                v = row[k]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cells.append(f"{float(v):>10.4f}")
                else:
                    cells.append(f"{str(v):>10s}")
            print(f"    {row['run_id']:>6d}  " + "  ".join(cells))
    print(f"  next:          {out['next_command']}")


# ──────────────────────────────────────────────────────────────────
# do_status
# ──────────────────────────────────────────────────────────────────


def do_status(params: dict[str, Any]) -> dict[str, Any]:
    wb = Workbook.open(Path(params["workbook"]))
    response = wb.response_name()
    all_runs = wb.all_runs()
    pending = wb.pending_runs()
    completed = wb.completed_runs()
    fitted = wb.read_parameters()
    fim_data = wb.read_fim()

    if pending:
        if completed:
            next_command = f"discopt doe fit {wb.path}  # {len(pending)} run(s) still pending"
        else:
            next_command = (
                f"# fill in '{response}' column for run_ids "
                f"{', '.join(str(r['run_id']) for r in pending)}, save, then: "
                f"discopt doe fit {wb.path}"
            )
    elif completed and not fitted:
        next_command = f"discopt doe fit {wb.path}"
    elif fitted:
        next_command = f"discopt doe extend {wb.path} --n N"
    else:
        next_command = f"discopt doe new ... -o {wb.path}"

    return {
        "workbook_path": str(wb.path),
        "template": wb.template_name(),
        "template_args": wb.template_args(),
        "module_callable": wb.module_callable(),
        "response_name": response,
        "input_specs": [s.to_dict() for s in wb.input_specs()],
        "n_total": len(all_runs),
        "n_completed": len(completed),
        "n_pending": len(pending),
        "next_batch_index": wb.next_batch_index(),
        "parameters": fitted,
        "has_fim": fim_data is not None,
        "fim_size": fim_data[0].shape[0] if fim_data is not None else 0,
        "next_command": next_command,
    }


def _cmd_status(args) -> int:
    try:
        out = do_status({"workbook": args.workbook})
    except (FileNotFoundError, ValueError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        label = out["template"] or out["module_callable"] or "(unknown model)"
        print(f"{out['workbook_path']}")
        print(f"  model:       {label}")
        inputs_str = ", ".join(f"{s['name']} in [{s['lb']}, {s['ub']}]" for s in out["input_specs"])
        print(f"  inputs:      {inputs_str}")
        print(f"  response:    {out['response_name']}")
        print(
            f"  runs:        {out['n_completed']} completed / "
            f"{out['n_pending']} pending / {out['n_total']} total"
        )
        if out["parameters"]:
            print("  parameters:")
            for p in out["parameters"]:
                est = p["estimate"]
                se = p["std_error"]
                if est is None:
                    continue
                se_s = f"{se:.4g}" if se is not None else "n/a"
                print(f"    {p['name']:>8s} = {est:>12.6g}  ± {se_s}")
        else:
            print("  parameters:  (not fit yet)")
        print(f"  next:        {out['next_command']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# do_fit
# ──────────────────────────────────────────────────────────────────


def do_fit(params: dict[str, Any]) -> dict[str, Any]:
    """Estimate parameters from the completed runs in a workbook.

    For the four built-in templates the model is linear in the
    parameters, so this is ordinary least squares: ``β̂ = (XᵀX)⁻¹ Xᵀy``
    with ``cov(β̂) = σ̂² (XᵀX)⁻¹``. The cumulative Fisher Information
    Matrix is ``Σᵢ FIM(dᵢ) = XᵀX / σ²`` and is written to the workbook
    for use as ``prior_fim`` by a subsequent ``extend``.

    Fitting for ``--module`` experiments is not yet implemented; the
    user should call :func:`discopt.estimate.estimate_parameters`
    directly until that path is wired through.
    """
    wb = Workbook.open(Path(params["workbook"]))
    response = wb.response_name()
    completed = wb.completed_runs()
    if not completed:
        raise DoEError(f"no completed runs in {wb.path}; fill in the '{response}' column first")

    template = wb.template_name()
    if not template:
        raise DoEError(
            "`discopt doe fit` for --module experiments is not yet implemented; "
            "use `discopt.estimate.estimate_parameters` directly in Python."
        )

    input_names = [s.name for s in wb.input_specs()]
    parameter_names = wb.rebuild_experiment()[1]
    sigma = wb.measurement_error()
    n_p = len(parameter_names)
    n_obs = len(completed)

    X = np.array(
        [
            _design_row(template, wb.template_args(), parameter_names, input_names, row)
            for row in completed
        ],
        dtype=np.float64,
    )
    y = np.array([float(row[response]) for row in completed], dtype=np.float64)

    if n_obs < n_p:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Undetermined fit — covariance / SE are not meaningful, but we
        # still return what we can so the user sees progress.
        residual_ss = float(np.sum((y - X @ beta) ** 2))
        cov = np.full((n_p, n_p), np.nan)
        std_errs = np.full(n_p, np.nan)
        cis_arr = np.full((n_p, 2), np.nan)
    else:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual_ss = float(np.sum((y - X @ beta) ** 2))
        dof = max(1, n_obs - n_p)
        sigma_hat_sq = residual_ss / dof if n_obs > n_p else sigma**2
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(X.T @ X)
        cov = sigma_hat_sq * xtx_inv
        std_errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        from scipy.stats import t as t_dist

        t_val = float(t_dist.ppf(0.975, df=dof)) if n_obs > n_p else float("nan")
        cis_arr = np.column_stack([beta - t_val * std_errs, beta + t_val * std_errs])

    estimates = {name: float(beta[i]) for i, name in enumerate(parameter_names)}
    std_errors = {name: float(std_errs[i]) for i, name in enumerate(parameter_names)}
    cis = {
        name: (float(cis_arr[i, 0]), float(cis_arr[i, 1])) for i, name in enumerate(parameter_names)
    }

    # FIM = XᵀX / σ² is the design's information about the parameters
    # under the user-declared measurement_error. Use the declared σ
    # (not σ̂) so the prior FIM reflects design quality rather than fit
    # residuals.
    cum_fim = (X.T @ X) / (sigma**2) + _RIDGE * np.eye(n_p)

    wb.write_parameters(parameter_names, estimates, std_errors, cis)
    wb.write_fim(cum_fim, parameter_names)
    coefficients, anova_rows, fit_summary = _compute_anova(
        y=y,
        X=X,
        beta=beta,
        residual_ss=residual_ss,
        std_errs=std_errs,
        parameter_names=parameter_names,
        n_obs=n_obs,
        n_p=n_p,
        sigma=sigma,
    )
    wb.write_anova(
        coefficients=coefficients,
        anova_rows=anova_rows,
        fit_summary=fit_summary,
    )
    wb.log("fit", {"n_completed": n_obs})
    wb.save()

    sign, logdet = np.linalg.slogdet(cum_fim)
    log_det_fim = float(logdet) if sign > 0 else float("-inf")

    return {
        "workbook_path": str(wb.path),
        "parameters": [
            {
                "name": name,
                "estimate": estimates[name],
                "std_error": std_errors[name],
                "ci_lower_95": cis[name][0],
                "ci_upper_95": cis[name][1],
            }
            for name in parameter_names
        ],
        "parameter_names": parameter_names,
        "n_observations": n_obs,
        "objective": residual_ss,
        "log_det_fim": log_det_fim,
        "next_command": f"discopt doe extend {wb.path} --n N",
    }


def _compute_anova(
    *,
    y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    residual_ss: float,
    std_errs: np.ndarray,
    parameter_names: list[str],
    n_obs: int,
    n_p: int,
    sigma: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, Any]]]:
    """Compute coefficient table, ANOVA decomposition, and fit summary.

    Assumes the model includes an intercept (true for every built-in
    template), so ``df_regression = n_p - 1``. Returns NaN-filled entries
    for fields that aren't statistically meaningful (e.g. when
    ``n_obs <= n_p``).
    """
    from scipy.stats import f as f_dist
    from scipy.stats import t as t_dist

    dof_resid = n_obs - n_p
    has_dof = dof_resid > 0
    nan = float("nan")

    if has_dof:
        t_crit = float(t_dist.ppf(0.975, df=dof_resid))
        t_stats = np.where(std_errs > 0, beta / np.where(std_errs > 0, std_errs, 1.0), nan)
        p_vals = np.where(
            std_errs > 0,
            2.0 * t_dist.sf(np.abs(t_stats), df=dof_resid),
            nan,
        )
    else:
        t_crit = nan
        t_stats = np.full(n_p, nan)
        p_vals = np.full(n_p, nan)

    coefficients: list[dict[str, Any]] = []
    for i, name in enumerate(parameter_names):
        se = float(std_errs[i])
        est = float(beta[i])
        ci_half = t_crit * se if has_dof and np.isfinite(se) else nan
        coefficients.append(
            {
                "name": name,
                "estimate": est,
                "std_error": se if np.isfinite(se) else nan,
                "t_statistic": float(t_stats[i]),
                "p_value": float(p_vals[i]),
                "ci_lower_95": est - ci_half if np.isfinite(ci_half) else nan,
                "ci_upper_95": est + ci_half if np.isfinite(ci_half) else nan,
            }
        )

    # Overall ANOVA (corrected total; intercept-aware).
    y_bar = float(np.mean(y))
    ss_total = float(np.sum((y - y_bar) ** 2))
    ss_residual = float(residual_ss)
    ss_regression = max(ss_total - ss_residual, 0.0)
    df_regression = max(n_p - 1, 0)
    df_total = n_obs - 1

    if has_dof and df_regression > 0:
        ms_regression = ss_regression / df_regression
        ms_residual = ss_residual / dof_resid
        f_stat = ms_regression / ms_residual if ms_residual > 0 else nan
        p_f = float(f_dist.sf(f_stat, df_regression, dof_resid)) if np.isfinite(f_stat) else nan
    else:
        ms_regression = nan
        ms_residual = nan
        f_stat = nan
        p_f = nan

    anova_rows: list[dict[str, Any]] = [
        {
            "source": "Regression",
            "ss": ss_regression,
            "df": df_regression,
            "ms": ms_regression,
            "f_statistic": f_stat,
            "p_value": p_f,
        },
        {
            "source": "Residual",
            "ss": ss_residual,
            "df": dof_resid,
            "ms": ms_residual,
            "f_statistic": None,
            "p_value": None,
        },
        {
            "source": "Total (corrected)",
            "ss": ss_total,
            "df": df_total,
            "ms": None,
            "f_statistic": None,
            "p_value": None,
        },
    ]

    if ss_total > 0:
        r2 = 1.0 - ss_residual / ss_total
        if has_dof:
            r2_adj = 1.0 - (1.0 - r2) * (n_obs - 1) / dof_resid
        else:
            r2_adj = nan
    else:
        r2 = nan
        r2_adj = nan
    rmse = float(np.sqrt(ms_residual)) if np.isfinite(ms_residual) else nan
    sigma_hat = rmse  # RMSE is the estimated σ from residuals

    fit_summary: list[tuple[str, Any]] = [
        ("n_observations", int(n_obs)),
        ("n_parameters", int(n_p)),
        ("degrees_of_freedom", int(dof_resid) if has_dof else "n/a (under-determined)"),
        ("R_squared", float(r2)),
        ("adjusted_R_squared", float(r2_adj)),
        ("RMSE (sigma_hat)", float(sigma_hat)),
        ("sigma_declared", float(sigma)),
        ("F_statistic", float(f_stat)),
        ("F_p_value", float(p_f)),
    ]
    return coefficients, anova_rows, fit_summary


def _design_row(
    template: str,
    template_args: dict[str, Any],
    parameter_names: list[str],
    input_names: list[str],
    row: dict[str, Any],
) -> np.ndarray:
    """Return the design-matrix row for one completed run.

    The entries are the basis functions of each parameter coefficient,
    evaluated at this run's inputs. The order matches ``parameter_names``.
    """
    xs = {nm: float(row[nm]) for nm in input_names}
    if template == "linear":
        return np.array([1.0] + [xs[nm] for nm in input_names], dtype=np.float64)
    if template == "polynomial-1d":
        x = xs[input_names[0]]
        degree = int(template_args.get("degree", len(parameter_names) - 1))
        return np.array([x**j for j in range(degree + 1)], dtype=np.float64)
    if template in ("response-surface-2d", "response-surface-3d"):
        n = len(input_names)
        vals = [xs[nm] for nm in input_names]
        cross = [vals[i] * vals[j] for i in range(n) for j in range(i + 1, n)]
        return np.array([1.0, *vals, *[v * v for v in vals], *cross], dtype=np.float64)
    if template in ("scheffe-linear", "scheffe-quadratic", "scheffe-special-cubic"):
        vals = [xs[nm] for nm in input_names]
        q = len(input_names)
        terms: list[float] = list(vals)
        if template == "scheffe-linear":
            return np.array(terms, dtype=np.float64)
        terms.extend(vals[i] * vals[j] for i in range(q) for j in range(i + 1, q))
        if template == "scheffe-quadratic":
            return np.array(terms, dtype=np.float64)
        terms.extend(
            vals[i] * vals[j] * vals[k]
            for i in range(q)
            for j in range(i + 1, q)
            for k in range(j + 1, q)
        )
        return np.array(terms, dtype=np.float64)
    raise DoEError(f"unknown template {template!r}")


def _cmd_fit(args) -> int:
    try:
        out = do_fit({"workbook": args.workbook})
    except (DoEError, FileNotFoundError, ValueError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"fit complete: {out['workbook_path']}")
        print(f"  observations: {out['n_observations']}")
        print(f"  objective:    {out['objective']:.6g}")
        print(f"  log-det FIM:  {out['log_det_fim']:.6g}")
        print("  parameters:")
        for p in out["parameters"]:
            est = p["estimate"]
            se = p["std_error"]
            if est is None:
                continue
            se_s = f"{se:.4g}" if se is not None else "n/a"
            print(f"    {p['name']:>8s} = {est:>12.6g}  ± {se_s}")
        print(f"  next:         {out['next_command']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# do_anova
# ──────────────────────────────────────────────────────────────────


def do_anova(params: dict[str, Any]) -> dict[str, Any]:
    """Run ANOVA on the completed runs of a Latin-family workbook."""
    from discopt.doe import anova_report

    wb = Workbook.open(Path(params["workbook"]))
    response = wb.response_name()
    completed = wb.completed_runs()
    if not completed:
        raise DoEError(f"no completed runs in {wb.path}; fill in '{response}' first")

    template = wb.template_name()
    template_args = wb.template_args()
    if template and template in {"latin-square", "graeco-latin", "hyper-graeco-latin"}:
        factors = list(template_args.get("levels", {}).keys())
    else:
        factors = [s.name for s in wb.input_specs()]

    include_replicate = params.get("include_replicate", False)
    interactions = params.get("interactions") or None
    rows: list[dict[str, Any]] = []
    for r in completed:
        d = {f: r[f] for f in factors if f in r}
        d[response] = r[response]
        if "replicate" in r and r["replicate"] is not None:
            d["replicate"] = r["replicate"]
        rows.append(d)

    table = anova_report(
        rows,
        response=response,
        factors=factors,
        interactions=interactions,
        include_replicate=include_replicate,
    )
    return {
        "workbook_path": str(wb.path),
        "response": response,
        "n_observations": table.n_obs,
        "grand_mean": table.grand_mean,
        "balanced": table.balanced,
        "rows": [
            {
                "source": r.source,
                "ss": r.ss,
                "df": r.df,
                "ms": r.ms,
                "f": r.f,
                "p": r.p,
            }
            for r in table.rows
        ],
        "summary": table.summary(),
    }


def _cmd_anova(args) -> int:
    try:
        interactions: list[tuple[str, ...]] | None = None
        if args.interaction:
            interactions = [tuple(s.split(":")) for s in args.interaction]
        out = do_anova(
            {
                "workbook": args.workbook,
                "include_replicate": args.include_replicate,
                "interactions": interactions,
            }
        )
    except (DoEError, FileNotFoundError, ValueError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(json.dumps({k: v for k, v in out.items() if k != "summary"}, indent=2))
    else:
        print(f"ANOVA on {out['workbook_path']}")
        print(f"  response:   {out['response']}")
        print(f"  n:          {out['n_observations']}")
        print(f"  grand mean: {out['grand_mean']:.6g}")
        print()
        print(out["summary"])
    return 0


# ──────────────────────────────────────────────────────────────────
# do_extend
# ──────────────────────────────────────────────────────────────────


def do_extend(params: ExtendParams) -> dict[str, Any]:
    from discopt.doe import batch_optimal_experiment

    wb = Workbook.open(params.workbook)
    experiment, parameter_names = wb.rebuild_experiment()
    input_specs = wb.input_specs()
    input_names = [s.name for s in input_specs]
    bounds = {s.name: (s.lb, s.ub) for s in input_specs}
    completed = wb.completed_runs()

    cached = wb.read_fim()
    if cached is not None and cached[1] == parameter_names:
        prior_fim = cached[0]
    else:
        prior_fim = _cumulative_fim_from_completed(
            experiment, parameter_names, completed, input_names
        )

    param_values = _param_values_for_design(parameter_names, wb)
    template = wb.template_name()
    template_args = wb.template_args()
    mixture_total = template_args.get("mixture_total")
    eq_cons, proj = _mixture_constraints(
        template,
        [(s.name, s.lb, s.ub) for s in input_specs],
        float(mixture_total) if mixture_total is not None else None,
    )
    if params.n == 1:
        from discopt.doe import optimal_experiment

        single = optimal_experiment(
            experiment,
            param_values,
            bounds,
            criterion=wb.criterion(),
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=wb.seed(),
        )
        designs = [single.design]
        criterion_value = float(single.criterion_value)
    else:
        batch = batch_optimal_experiment(
            experiment,
            param_values,
            bounds,
            n_experiments=params.n,
            criterion=wb.criterion(),
            prior_fim=prior_fim,
            equality_constraints=eq_cons,
            feasible_projection=proj,
            n_starts=params.n_starts,
            seed=wb.seed(),
        )
        designs = list(batch.designs)
        criterion_value = float(batch.criterion_value)

    batch_idx = wb.next_batch_index()
    new_ids = wb.append_runs(batch_idx, designs)
    wb.log("extend", {"n": params.n, "batch": batch_idx})
    wb.save()

    return {
        "workbook_path": str(wb.path),
        "batch": batch_idx,
        "new_run_ids": new_ids,
        "designs": [
            {"run_id": rid, **{nm: float(d[nm]) for nm in input_names}}
            for rid, d in zip(new_ids, designs)
        ],
        "criterion": wb.criterion(),
        "criterion_value": criterion_value,
        "parameter_names": parameter_names,
        "next_command": f"discopt doe status {wb.path}",
    }


def _cmd_extend(args) -> int:
    try:
        out = do_extend(
            ExtendParams(
                workbook=Path(args.workbook),
                n=int(args.n),
                n_starts=int(args.n_starts),
            )
        )
    except (DoEError, FileNotFoundError, ValueError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"extended workbook: {out['workbook_path']}")
        new_ids = out["new_run_ids"]
        print(f"  batch:        {out['batch']}  (run_ids {min(new_ids)}-{max(new_ids)})")
        print(f"  criterion:    {out['criterion']} = {out['criterion_value']:.6g}")
        print("  new runs:")
        if out["designs"]:
            keys = [k for k in out["designs"][0].keys() if k != "run_id"]
            print("    run_id  " + "  ".join(f"{k:>10s}" for k in keys))
            for row in out["designs"]:
                cells = [f"{row[k]:>10.4f}" for k in keys]
                print(f"    {row['run_id']:>6d}  " + "  ".join(cells))
        print(f"  next:         {out['next_command']}")
    return 0


# ──────────────────────────────────────────────────────────────────
# optimize
# ──────────────────────────────────────────────────────────────────


def _parse_kv_float_eq(s: str) -> tuple[str, float]:
    """Parse a ``key=value`` string with a float value."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {s!r}")
    k, _, v = s.partition("=")
    try:
        return k.strip(), float(v)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"value for {k!r} is not a float: {v!r}") from e


def _cmd_optimize(args) -> int:
    try:
        acq_kwargs: dict[str, float] | None = None
        if getattr(args, "acquisition_kwarg", None):
            acq_kwargs = dict(args.acquisition_kwarg)
        custom_kwargs: dict[str, Any] | None = None
        if getattr(args, "surrogate_kwarg", None):
            custom_kwargs = {k: v for k, v in args.surrogate_kwarg}
        out = do_optimize(
            OptimizeParams(
                workbook=Path(args.workbook),
                criterion=args.criterion,
                surrogate=args.surrogate,
                acquisition=args.acquisition,
                batch_size=int(args.batch_size),
                n_candidates=int(args.n_candidates),
                seed=args.seed,
                acquisition_kwargs=acq_kwargs,
                custom_surrogate_path=args.custom_surrogate,
                custom_surrogate_kwargs=custom_kwargs,
            )
        )
    except (DoEError, FileNotFoundError, ValueError) as e:
        return _fail(args, str(e), workbook_path=args.workbook)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"optimized workbook: {out['workbook_path']}")
        print(
            f"  criterion:    {out['criterion']}   acquisition: {out['acquisition']}   "
            f"surrogate: {out['surrogate']} ({out['surrogate_mode']})"
        )
        print(f"  completed:    {out['n_completed']} runs")
        if out["incumbent_x"] is not None:
            x_str = ", ".join(f"{k}={v:.4g}" for k, v in out["incumbent_x"].items())
            print(f"  incumbent:    y={out['incumbent_y']:.6g} at ({x_str})")
        new_ids = out["new_run_ids"]
        if new_ids:
            print(f"  batch:        {len(new_ids)}  (run_ids {min(new_ids)}-{max(new_ids)})")
            print("  new runs:")
            keys = [k for k in out["next_designs"][0].keys() if k != "run_id"]
            print("    run_id  " + "  ".join(f"{k:>10s}" for k in keys))
            for row in out["next_designs"]:
                cells = [
                    f"{row[k]:>10.4g}" if isinstance(row[k], (int, float)) else f"{row[k]:>10s}"
                    for k in keys
                ]
                print(f"    {row['run_id']:>6d}  " + "  ".join(cells))
    return 0


# ──────────────────────────────────────────────────────────────────
# gui
# ──────────────────────────────────────────────────────────────────


def _cmd_gui(args) -> int:
    from discopt.doe.gui.launcher import launch

    return launch(
        workbook=args.workbook,
        port=args.port,
        open_browser=not args.no_browser,
    )


# ──────────────────────────────────────────────────────────────────
# error formatting
# ──────────────────────────────────────────────────────────────────


def _fail(args, msg: str, *, workbook_path: str | None = None) -> int:
    if getattr(args, "json", False):
        payload: dict[str, Any] = {"error": msg}
        if workbook_path is not None:
            payload["workbook_path"] = workbook_path
        print(json.dumps(payload, indent=2), file=sys.stdout)
    else:
        print(f"error: {msg}", file=sys.stderr)
    return 1


# ──────────────────────────────────────────────────────────────────
# argparse registration
# ──────────────────────────────────────────────────────────────────


def add_subparser(subparsers) -> None:
    """Register the ``doe`` subcommand on the top-level ``discopt`` parser."""
    p = subparsers.add_parser(
        "doe",
        help="Design of experiments with Excel-workbook campaigns",
        description=(
            "Optimal experimental design. Verbs: templates, new, status, "
            "fit, anova, extend, optimize, gui. Each accepts --json for "
            "machine-readable output."
        ),
    )
    doe_sub = p.add_subparsers(dest="doe_cmd", metavar="<verb>", required=True)

    # --- templates ---
    p_tmpl = doe_sub.add_parser("templates", help="List built-in experiment templates.")
    _add_json(p_tmpl)
    p_tmpl.set_defaults(doe_func=_cmd_templates)

    # --- new ---
    p_new = doe_sub.add_parser("new", help="Create a workbook with N optimal initial runs.")
    new_sub = p_new.add_subparsers(dest="template", metavar="<template>", required=True)

    from discopt.doe.templates import COMBINATORIAL_TEMPLATES

    for tmpl in TEMPLATE_NAMES:
        sp = new_sub.add_parser(tmpl, help=_TEMPLATE_DESCRIPTIONS[tmpl])
        if tmpl == "factorial-2level":
            sp.add_argument(
                "--factor",
                action="append",
                type=_parse_factor_pair,
                required=True,
                help="Factor as NAME:LOW:HIGH (repeatable; LOW/HIGH numeric or string).",
            )
            sp.add_argument(
                "--center-points",
                type=int,
                default=0,
                help="Center-point runs per replicate (numeric factors only; default 0).",
            )
            sp.add_argument(
                "--replicates",
                type=int,
                default=1,
                help="Whole-design replications (default 1).",
            )
        elif tmpl in COMBINATORIAL_TEMPLATES:
            sp.add_argument(
                "--levels",
                action="append",
                type=_parse_levels_spec,
                required=True,
                help="Factor levels as NAME:L1,L2,... (repeatable).",
            )
            sp.add_argument(
                "--replicates",
                type=int,
                default=1,
                help="Whole-design replications (default 1).",
            )
        else:
            sp.add_argument(
                "--input",
                action="append",
                type=_parse_input_spec,
                required=True,
                help="Design factor as NAME:LB:UB (repeatable).",
            )
        if tmpl == "polynomial-1d":
            sp.add_argument("--degree", type=int, required=True, help="Polynomial degree (>= 1).")
        if tmpl == "optimize":
            sp.add_argument(
                "--optimize-criterion",
                default="maximize",
                choices=("maximize", "minimize"),
                help="Active-learning direction (default 'maximize').",
            )
            sp.add_argument(
                "--optimize-surrogate",
                default="gp",
                help="Surrogate preset (default 'gp'); see SURROGATE_PRESETS.",
            )
            sp.add_argument(
                "--optimize-acquisition",
                default="expected_improvement",
                help="Acquisition (default 'expected_improvement').",
            )
        if tmpl.startswith("scheffe-"):
            sp.add_argument(
                "--mixture-total",
                type=float,
                default=1.0,
                help="Required sum of the component values (default 1.0).",
            )
        _add_common_new_options(sp)
        sp.set_defaults(doe_func=_cmd_new, _is_module=False, bounds=None, params=None, module=None)

    # Escape-hatch: --module
    p_module = new_sub.add_parser(
        "module",
        help="Use a user-defined Experiment loaded via 'pkg.mod:callable'.",
    )
    p_module.add_argument(
        "--module",
        required=True,
        help="Spec of the form 'pkg.mod:callable' returning an Experiment.",
    )
    p_module.add_argument(
        "--bounds",
        action="append",
        type=_parse_input_spec,
        required=True,
        help="Design factor as NAME:LB:UB (repeatable).",
    )
    p_module.add_argument(
        "--params",
        action="append",
        type=_parse_kv_float,
        required=True,
        help="Prior parameter value as NAME=VALUE (repeatable).",
    )
    _add_common_new_options(p_module)
    p_module.set_defaults(doe_func=_cmd_new, _is_module=True, input=None, degree=None)

    # --- status ---
    p_status = doe_sub.add_parser("status", help="Show campaign state and suggested next step.")
    p_status.add_argument("workbook", help="Path to the .xlsx workbook.")
    _add_json(p_status)
    p_status.set_defaults(doe_func=_cmd_status)

    # --- fit ---
    p_fit = doe_sub.add_parser("fit", help="Estimate parameters from completed runs.")
    p_fit.add_argument("workbook", help="Path to the .xlsx workbook.")
    _add_json(p_fit)
    p_fit.set_defaults(doe_func=_cmd_fit)

    # --- anova ---
    p_anova = doe_sub.add_parser(
        "anova",
        help="Run ANOVA on a completed Latin-family workbook.",
    )
    p_anova.add_argument("workbook", help="Path to the .xlsx workbook.")
    p_anova.add_argument(
        "--interaction",
        action="append",
        default=None,
        help="Interaction term as 'A:B' or 'A:B:C' (repeatable).",
    )
    p_anova.add_argument(
        "--include-replicate",
        action="store_true",
        help="Treat the replicate column as a blocking factor.",
    )
    _add_json(p_anova)
    p_anova.set_defaults(doe_func=_cmd_anova)

    # --- optimize ---
    p_opt = doe_sub.add_parser(
        "optimize",
        help="Run one active-learning round on an optimize-template workbook.",
        description=(
            "Fit the chosen surrogate to completed runs, score Sobol "
            "candidates with the chosen acquisition, and append the "
            "recommended batch as pending runs."
        ),
    )
    p_opt.add_argument("workbook", help="Path to the .xlsx workbook (template=optimize).")
    p_opt.add_argument(
        "--criterion",
        default="maximize",
        choices=("maximize", "minimize"),
        help="Optimization direction (default 'maximize').",
    )
    p_opt.add_argument(
        "--surrogate",
        default="gp",
        help="Surrogate preset name (e.g. 'gp', 'rf', 'linear'); see SURROGATE_PRESETS.",
    )
    p_opt.add_argument(
        "--custom-surrogate",
        default=None,
        help=(
            "Dotted import path of a custom surrogate class, e.g. "
            "'mypkg.bayes:MyBayes'. Overrides --surrogate when given."
        ),
    )
    p_opt.add_argument(
        "--surrogate-kwarg",
        action="append",
        type=_parse_kv_float_eq,
        default=None,
        help="kwarg KEY=VALUE for --custom-surrogate constructor (repeatable).",
    )
    p_opt.add_argument(
        "--acquisition",
        default="expected_improvement",
        help="Acquisition function: 'expected_improvement', 'confidence_bound', 'steepest_ascent'.",
    )
    p_opt.add_argument(
        "--acquisition-kwarg",
        action="append",
        type=_parse_kv_float_eq,
        default=None,
        help="kwarg KEY=VALUE for the acquisition (e.g. beta=2.0); repeatable.",
    )
    p_opt.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of new runs to recommend (default 4).",
    )
    p_opt.add_argument(
        "--n-candidates",
        type=int,
        default=2048,
        help="Sobol candidate pool size (default 2048).",
    )
    p_opt.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducible candidate-sampling seed.",
    )
    _add_json(p_opt)
    p_opt.set_defaults(doe_func=_cmd_optimize)

    # --- extend ---
    p_ext = doe_sub.add_parser(
        "extend", help="Append M more optimal runs using the cumulative FIM as prior."
    )
    p_ext.add_argument("workbook", help="Path to the .xlsx workbook.")
    p_ext.add_argument("--n", type=int, required=True, help="Number of additional runs.")
    p_ext.add_argument(
        "--n-starts",
        type=int,
        default=10,
        help="Multi-start budget for each single-design search (default 10).",
    )
    _add_json(p_ext)
    p_ext.set_defaults(doe_func=_cmd_extend)

    # --- gui ---
    p_gui = doe_sub.add_parser(
        "gui",
        help="Launch the Streamlit GUI over a workbook (requires discopt[doe-gui]).",
    )
    p_gui.add_argument(
        "workbook",
        nargs="?",
        default=None,
        help="Optional path to an existing .xlsx workbook to open on startup.",
    )
    p_gui.add_argument(
        "--port",
        type=int,
        default=None,
        help="TCP port to bind on 127.0.0.1 (default: pick a free port).",
    )
    p_gui.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the default browser automatically.",
    )
    p_gui.set_defaults(doe_func=_cmd_gui)


def _add_common_new_options(sp) -> None:
    sp.add_argument("-o", "--output", required=True, help="Output .xlsx path.")
    sp.add_argument("--n", type=int, default=1, help="Number of initial runs (default 1).")
    sp.add_argument(
        "--response",
        default="y",
        help="Name of the response column in the workbook (default 'y').",
    )
    sp.add_argument(
        "--error",
        type=float,
        default=1.0,
        help="Measurement error stdev (default 1.0).",
    )
    sp.add_argument(
        "--criterion",
        default=_DEFAULT_CRITERION,
        choices=(*_CRITERION_CHOICES, *_CRITERION_ALIASES.keys()),
        help="Optimality criterion (default determinant aka D).",
    )
    sp.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    sp.add_argument(
        "--n-starts",
        type=int,
        default=10,
        help="Multi-start budget for each single-design search (default 10).",
    )
    sp.add_argument("--force", action="store_true", help="Overwrite existing output file.")
    _add_json(sp)


def _add_json(sp) -> None:
    sp.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )


def run(args) -> int:
    """Dispatch ``discopt doe ...`` after argparse parsing."""
    func = getattr(args, "doe_func", None)
    if func is None:
        print("no doe subcommand given; try `discopt doe --help`", file=sys.stderr)
        return 1
    return int(func(args) or 0)


__all__ = [
    "DoEError",
    "ExtendParams",
    "NewParams",
    "add_subparser",
    "do_anova",
    "do_extend",
    "do_fit",
    "do_new",
    "do_status",
    "do_templates",
    "run",
]
