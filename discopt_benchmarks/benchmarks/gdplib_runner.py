"""GDPlib benchmark runner (issue #823).

Runs discopt against the SECQUOIA GDPlib corpus of Generalized Disjunctive
Programming models (https://github.com/SECQUOIA/gdplib). GDPlib ships *Pyomo GDP*
models (``Disjunct`` / ``Disjunction`` / ``LogicalConstraint``), which are not a
file format discopt reads directly. The bridge is:

    build_model()                    # gdplib -> Pyomo GDP model
    TransformationFactory('gdp.bigm' | 'gdp.hull').apply_to(m)   # GDP -> MI(N)LP
    SolverFactory('discopt').solve(m)                            # discopt.pyomo

The Pyomo ``gdp.bigm`` / ``gdp.hull`` transformations lower the disjunctions to a
standard mixed-integer (non)linear program, which the existing ``discopt.pyomo``
plugin round-trips through a temporary ``.nl`` file and solves in-process. This
runner automates that pipeline over the corpus and emits ``metrics.SolveResult``
objects so GDPlib results flow through the same correctness/reporting pipeline as
the MINLPLib and CUTEst runners.

Correctness (the non-negotiable gate, per ``CLAUDE.md``): where an independent
oracle exists we cross-check discopt's certified objective against it and count
any disagreement as *incorrect*. **Every oracle value is feasibility-verified** —
trusted only when the solver proved optimality *and* its incumbent, evaluated in
the real pyomo model, satisfies every constraint (:func:`_max_constraint_violation`).
This is deliberate: a claimed optimum below the true optimum is exactly a claimed
feasible point that isn't feasible, and an unverified one can *mask* a discopt false
primal — the hole that let the old pyscipopt-``.nl`` path certify a below-true cstr
optimum (#823). Oracles, most-trusted first: **HiGHS** for the linear (MILP) subset
(exact); **SCIP and BARON via GAMS** for the nonlinear subset (a solution-loadable,
verifiable path — when both prove optimality they must agree, else neither is
trusted); and the BARON-confirmed :func:`reference_optima` table as an offline
fallback. A discopt objective *strictly better* than the oracle optimum (an
infeasible incumbent / false primal), a claimed-optimal objective that disagrees, or
a dual bound on the wrong side of the optimum, is a soundness violation surfaced
loudly — never masked.

Requirements (``pip install discopt-benchmarks[gdplib]``): ``pyomo`` and
``gdplib``. Install ``gdplib`` **from source** — the PyPI wheel omits the model
data files (``.dat`` / ``.xlsx`` / ``.txt``), so most builders raise
``FileNotFoundError`` under it::

    pip install "gdplib @ git+https://github.com/SECQUOIA/gdplib.git"

The oracles are optional: the linear subset needs ``highspy`` (``appsi_highs``); the
nonlinear subset needs **GAMS** (with SCIP/BARON subsolvers). Absent them, checks
fall back to the :func:`reference_optima` table (which is itself BARON-confirmed).
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from benchmarks.metrics import (
    DISCOPT_STATUS_MAP,
    BenchmarkResults,
    InstanceInfo,
    SolveResult,
    SolveStatus,
    dual_bound_crosses_optimum,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# discopt status string -> benchmark SolveStatus. The shared table plus the
# Pyomo/GAMS spellings this runner also sees (``maxtimelimit``/``maxiterations``).
_STATUS_MAP = DISCOPT_STATUS_MAP | {
    # Pyomo/GAMS spellings this runner also sees.
    "maxtimelimit": SolveStatus.TIME_LIMIT,
    "maxiterations": SolveStatus.TIME_LIMIT,
}


@dataclass
class GDPModelSpec:
    """A discovered GDPlib model: its name and zero-arg builder."""

    name: str
    builder: Callable[[], object]
    module: str


@dataclass
class GDPLibSuiteConfig:
    """Configuration for a GDPlib benchmark sweep."""

    name: str = "gdplib"
    description: str = ""
    methods: tuple[str, ...] = ("bigm",)  # gdp reformulations to run
    time_limit_seconds: float = 300.0
    max_variables: int | None = None  # skip models whose reformulation exceeds
    include: list[str] | None = None  # explicit model-name allowlist
    exclude: list[str] = field(default_factory=list)
    oracle: bool = True  # cross-check the linear subset against HiGHS


#: The curated small set: every GDPlib model whose ``gdp.bigm`` reformulation fits
#: under 500 variables, in increasing size. This is the panel the GDP work is
#: scored on (#823, #993), so it is pinned by NAME rather than derived from
#: ``--max-variables`` at run time. Two reasons, both learned here: the sizes are
#: a property of the installed gdplib revision, so a size filter silently changes
#: the population between installs and makes two panels incomparable; and a
#: threshold quietly admits a model nothing has ever verified. Sizes are recorded
#: in ``docs/dev/gdplib-benchmarking.md``.
#:
#: ``multiperiod_blending`` (474 vars) is deliberately absent: it fits the size
#: bound but no in-repo oracle proves its optimum, so it can contribute a timing
#: number and nothing to a soundness check. Add it here the day it has a verified
#: reference.
GDPLIB_SMALL: tuple[str, ...] = (
    "jobshop",
    "ex1_linan_2023",
    "positioning",
    "small_batch",
    "cstr",
    "spectralog",
    "methanol",
    "batch_processing",
    "syngas",
    "water_network",
    "gdp_col",
    "modprodnet",
)

#: GDPlib's own benchmark panel (``gdplib/benchmark.py:PR58_BENCHMARK_INSTANCES``),
#: the GDPlib half of the #1775 parity panel (spec §1.2).
GDPLIB_PR58: tuple[str, ...] = (
    "batch_processing",
    "biofuel",
    "cstr",
    "disease_model",
    "ex1_linan_2023",
    "gdp_col",
    "hda",
    "kaibel",
    "med_term_purchasing",
    "methanol",
    "mod_hens",
    "modprodnet",
    "positioning",
    "small_batch",
    "spectralog",
    "stranded_gas",
    "syngas",
)

#: The #1775 GDP parity panel (spec §1.2): GDPlib's PR58 set plus the Pyomo GDPopt
#: test models, vendored from Pyomo 6.10.0 (``benchmarks.pyomo_gdp_examples``):
#: 8PP in its algebraic and logical forms, strip packing, constrained layout,
#: ``ex_633_trespalacios``, ``nonconvex_HEN`` and jobshop (``jobshop-small.dat``, the
#: data the example's own ``__main__`` uses). Every name has a row in
#: :func:`panel_oracle`: a verified optimum or an explicit "unproven".
GDP_PARITY: tuple[str, ...] = GDPLIB_PR58 + (
    "pyomo.eight_process",
    "pyomo.eight_process_logical",
    "pyomo.strip_packing_concrete",
    "pyomo.constrained_layout_clay0203",
    "pyomo.ex_633_trespalacios",
    "pyomo.nonconvex_hen",
    "pyomo.jobshop_small",
)

#: Named model sets selectable with ``--suite``.
SUITES: dict[str, tuple[str, ...]] = {"gdplib_small": GDPLIB_SMALL, "gdp_parity": GDP_PARITY}


def is_available() -> bool:
    """True if Pyomo and GDPlib are both importable."""
    try:
        import gdplib  # noqa: F401
        import pyomo  # noqa: F401

        return True
    except ImportError:
        return False


def discover_models(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[GDPModelSpec]:
    """Enumerate discoverable GDPlib models as ``(name, builder)`` specs.

    Walks the ``gdplib`` namespace and collects every zero-argument ``build_model``
    (and, for the multi-model subpackages, every other ``build_*`` that takes no
    required arguments). Submodules that fail to *import* — e.g. a data file the
    PyPI wheel omitted, or a Pyomo-version incompatibility — are skipped silently
    here (they surface as errors only if explicitly requested via ``include``).

    No model is filtered by name: a builder that needs an external tool (ipopt /
    GAMS) at build time is still listed, and :func:`solve_model` records its build
    failure as an ``ERROR`` run rather than hiding it — honest reporting over a
    hardcoded, environment-specific skip list.
    """
    import gdplib

    exclude_set = set(exclude or [])
    specs: list[GDPModelSpec] = []
    seen: set[str] = set()
    for mod_info in pkgutil.iter_modules(gdplib.__path__):
        mod_name = mod_info.name
        try:
            sub = importlib.import_module(f"gdplib.{mod_name}")
        except Exception:
            continue
        for attr in dir(sub):
            if not attr.startswith("build"):
                continue
            fn = getattr(sub, attr)
            if not callable(fn) or not _is_zero_arg(fn):
                continue
            # Prefer the short module name when the builder is the canonical
            # ``build_model``; otherwise disambiguate as ``module.builder``.
            name = mod_name if attr == "build_model" else f"{mod_name}.{attr}"
            if name in seen:
                continue
            seen.add(name)
            specs.append(GDPModelSpec(name=name, builder=fn, module=mod_name))

    if include is not None:
        want = set(include)
        specs = [s for s in specs if s.name in want]
        # #1775 G0.5: the vendored Pyomo examples are named ``pyomo.<name>`` and
        # resolved only when asked for by name, so the default sweep is unchanged.
        # An unknown ``pyomo.`` name is dropped like any other absent name; the
        # --suite path refuses that loudly.
        from benchmarks import pyomo_gdp_examples as px

        for name in sorted(want):
            if name.startswith(px.PREFIX) and name[len(px.PREFIX) :] in px.CATALOG:
                specs.append(
                    GDPModelSpec(name=name, builder=px.builder(name), module="pyomo_gdp_examples")
                )
    if exclude_set:
        specs = [s for s in specs if s.name not in exclude_set]
    return sorted(specs, key=lambda s: s.name)


def _is_zero_arg(fn: Callable) -> bool:
    """True if *fn* can be called with no arguments (all params have defaults)."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True  # builtins / C-callables: assume callable and let build fail
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            return False
    return True


def reference_optima() -> dict[str, float]:
    """Verified reference optima for GDPlib models, keyed by ``name/method``.

    Seeded only with values independently confirmed in-repo (e.g. via the HiGHS
    oracle on a linear model, or a value stable across bigm/hull and matching the
    GDPlib documentation). Used by :func:`assess_correctness` as the oracle for
    models HiGHS cannot check (nonlinear). Extend this as values are verified — a
    seed here is a *regression anchor*, not a target to tune toward.

    Every entry is **independently confirmed by BARON** (via GAMS, ``optcr=0``,
    ``lb == ub``) as well as SCIP, and a GDP's optimum is reformulation-independent,
    so the same value anchors both big-M and hull. Models neither solver proves
    within budget (methanol, gdp_col) are deliberately absent — an unproven incumbent
    is not a certified optimum and must never seed an oracle.

    **cstr correction (2026-07-22):** the value here was originally ``3.0543118``,
    taken from the old pyscipopt-``.nl`` oracle path. That path returned a *false
    optimum* for cstr — ``3.0543 < 3.0620``, i.e. **below** the true minimum —
    apparently solving a mis-encoded/relaxed model yet reporting ``gap = 0``. Two
    independent solvers via a solution-loadable path (BARON and SCIP, both through
    GAMS) prove ``3.0620`` with a pyomo-verified feasible point (max constraint
    violation ~1e-6), and discopt's own incumbent agrees. The seed is corrected to the
    BARON-proven value. Lesson: a single solver's "proven optimal" through a lossy file
    path is not trustworthy — which is why the oracle now feasibility-verifies every
    incumbent and routes the global check through GAMS (see :func:`_attach_oracle`).
    """
    return {
        # jobshop is also HiGHS-checkable (linear); the rest are nonlinear GDPs.
        "jobshop": 11.0,
        "ex1_linan_2023": -0.9995999999999999,
        "positioning": -8.064136166293226,
        "small_batch": 167427.6515668371,
        "cstr": 3.0620073,  # BARON-proven; pyscipopt-.nl gave a false 3.0543 (see docstring)
        "spectralog": 12.089261322767793,
        "syngas": 4669.0234827946,
        "water_network": 348337.03671302047,
        "modprodnet": 3592.924373781839,
        "batch_processing": 679365.33,  # BARON-certified (SCIP left it unproven at 60s)
    }


@dataclass(frozen=True)
class OracleRecord:
    """The oracle status of one parity-panel instance (#1775 G0.5).

    ``source`` says what certified ``value``:

    * ``"baron+scip"`` / ``"baron"`` -- the :func:`reference_optima` entries
      (global solvers through GAMS, ``lb == ub``, feasibility-verified point);
    * ``"highs"`` -- a linear instance, solved by HiGHS to proven optimality on
      **both** the ``gdp.bigm`` and the ``gdp.hull`` reformulation with the
      incumbent feasibility-verified in Pyomo each time (:func:`_solve_with_highs`),
      and the two values agreeing;
    * ``"unproven"`` -- no oracle yet; ``value`` is ``None`` and ``note`` says why.
      ``discopt_benchmarks/scripts/gdp_parity_oracle.py`` fills these from the
      BARON+SCIP GAMS oracle where GAMS is installed.

    ``pyomo_expected`` is Pyomo's own test-suite value, a cross-check and never an
    oracle (``pyomo_gdp_examples.PYOMO_EXPECTED``).
    """

    value: float | None
    source: str
    note: str = ""
    pyomo_expected: tuple[float, float] | None = None


#: HiGHS-certified optima of the linear parity-panel instances (see
#: :class:`OracleRecord`); measured 2026-10-19 with HiGHS through ``appsi_highs``.
#: The value is the ``gdp.bigm`` one. Re-measured 2026-10-19 (#1802) with the oracle
#: now judging each incumbent at its integral rounding: ``gdp.hull`` agrees to
#: <= 3.2e-13 relative on every row (disease_model 3.1e-13, the rest <= 3.3e-16), and
#: each value below is within 3.2e-13 of that re-measurement. (Earlier figures of
#: "<= 4e-14" here and "<= 6e-15" in the PR text were both under the measured value.)
_HIGHS_OPTIMA: dict[str, float] = {
    "disease_model": 304.4162107198232,
    "med_term_purchasing": 6797.539701513444,
    "pyomo.jobshop_small": 11.0,
    "pyomo.strip_packing_concrete": 11.0,  # Pyomo's test asserts 11 too
}

#: Nonlinear parity-panel instances certified by the GAMS oracle
#: (``scripts/gdp_parity_oracle.py``): GAMS 53, ``gdp.bigm``, ``optcr=1e-9``,
#: ``reslim=300`` s per solver, BARON on the full (Floating Network) license, run
#: 2026-10-19 (#1802). Each value closed its gap (``lb == ub``) with the incumbent
#: feasible in Pyomo at its integral rounding (:data:`_ORACLE_INT_TOL`). The source
#: says who certified: ``"baron+scip"`` agreed; ``"baron"`` alone where GAMS/SCIP's
#: point failed the rounding check -- on both 8PP forms SCIP reported 67.3236, below
#: the optimum, on binaries at 9.5e-7 / 0.999999 (#1802 S2).
_GAMS_OPTIMA: dict[str, tuple[float, str]] = {
    "pyomo.eight_process": (68.00974405106514, "baron"),
    "pyomo.eight_process_logical": (68.00973353554782, "baron"),
    "pyomo.constrained_layout_clay0203": (41573.26252165382, "baron+scip"),
    "pyomo.ex_633_trespalacios": (4.460367612328062, "baron+scip"),
    "pyomo.nonconvex_hen": (114384.77758268737, "baron+scip"),
}

_GAMS_300S = "GAMS oracle (2026-10-19, gdp.bigm, 300 s per solver) did not certify it: "

#: Panel instances with no oracle yet, and why (the declines the oracle script
#: printed for that run).
_UNPROVEN: dict[str, str] = {
    "biofuel": _GAMS_300S + "BARON gap open (92.16 .. 4066.17); SCIP hit the limit",
    "gdp_col": _GAMS_300S + "BARON gap open (9802.14 .. 19548.72); SCIP hit the limit",
    "hda": _GAMS_300S + "neither solver returned bounds within the limit",
    "kaibel": _GAMS_300S + "neither solver returned bounds within the limit; "
    "its builder also needs an ipopt executable",
    "methanol": _GAMS_300S + "BARON gap open (-2425.48 .. -1793.43), SCIP gap open",
    "mod_hens": _GAMS_300S + "BARON gap open (89586.38 .. 106766.53), SCIP gap open; "
    "its builder also needs an ipopt executable",
    "stranded_gas": _GAMS_300S + "neither solver returned bounds within the limit",
}


def panel_oracle() -> dict[str, OracleRecord]:
    """One :class:`OracleRecord` per :data:`GDP_PARITY` instance (#1775 G0.5).

    Every panel name has a row: a certified value, or ``"unproven"`` with a reason.
    A test pins that, so the panel cannot grow an instance nobody has an oracle
    story for.
    """
    from benchmarks.pyomo_gdp_examples import PREFIX, PYOMO_EXPECTED

    ref = reference_optima()
    out: dict[str, OracleRecord] = {}
    for name in GDP_PARITY:
        expected = PYOMO_EXPECTED.get(name[len(PREFIX) :]) if name.startswith(PREFIX) else None
        if name in ref:
            source = "baron" if name == "batch_processing" else "baron+scip"
            out[name] = OracleRecord(ref[name], source, pyomo_expected=expected)
        elif name in _HIGHS_OPTIMA:
            out[name] = OracleRecord(_HIGHS_OPTIMA[name], "highs", pyomo_expected=expected)
        elif name in _GAMS_OPTIMA:
            value, source = _GAMS_OPTIMA[name]
            out[name] = OracleRecord(value, source, pyomo_expected=expected)
        elif name in _UNPROVEN:
            out[name] = OracleRecord(None, "unproven", _UNPROVEN[name], pyomo_expected=expected)
    return out


def _classify(pyomo_model) -> tuple[InstanceInfo, bool]:
    """Return ``(InstanceInfo, is_linear)`` for a *reformulated* Pyomo model.

    ``is_linear`` drives oracle selection: HiGHS is a valid, exact oracle only for
    a fully linear (MILP) reformulation.
    """
    import pyomo.environ as pyo
    from pyomo.repn import generate_standard_repn

    n_var = n_int = n_bin = n_cont = 0
    for v in pyomo_model.component_data_objects(pyo.Var, active=True):
        n_var += 1
        if v.is_binary():
            n_bin += 1
        elif v.is_integer():
            n_int += 1
        else:
            n_cont += 1

    n_con = 0
    n_nonlin = 0
    is_linear = True
    for c in pyomo_model.component_data_objects(pyo.Constraint, active=True):
        n_con += 1
        repn = generate_standard_repn(c.body, quadratic=True)
        if not repn.is_linear():
            n_nonlin += 1
            is_linear = False
    for obj in pyomo_model.component_data_objects(pyo.Objective, active=True):
        if not generate_standard_repn(obj.expr, quadratic=True).is_linear():
            is_linear = False
        break

    info = InstanceInfo(
        name="",  # filled by caller (includes method suffix)
        num_variables=n_var,
        num_constraints=n_con,
        num_integer_vars=n_int,
        num_binary_vars=n_bin,
        num_continuous_vars=n_cont,
        num_nonlinear_constraints=n_nonlin,
        problem_class="gdp-linear" if is_linear else "gdp-nonlinear",
        is_convex=None,
        source="gdplib",
    )
    return info, is_linear


def _objective_value(pyomo_model) -> float | None:
    import pyomo.environ as pyo
    from pyomo.core.expr import value as pyo_value

    for obj in pyomo_model.component_data_objects(pyo.Objective, active=True):
        try:
            return float(pyo_value(obj.expr))
        except Exception:
            return None
    return None


def _is_minimize(pyomo_model) -> bool:
    import pyomo.environ as pyo

    for obj in pyomo_model.component_data_objects(pyo.Objective, active=True):
        return obj.sense == 1  # pyo.minimize
    return True


@dataclass
class ModelRun:
    """One (model, method) run: the discopt result plus oracle cross-check."""

    name: str  # ``<model>/<method>``
    info: InstanceInfo
    discopt: SolveResult
    is_linear: bool
    minimize: bool
    oracle_objective: float | None = None
    oracle_source: str | None = None  # "highs" | "scip" | "reference" | None
    # True when the objective came from a point that ended at a non-feasible
    # status (e.g. ``time_limit``) and was feasibility-verified before reporting.
    # Such an incumbent must face the same oracle cross-check as any other.
    incumbent_verified: bool = False
    # Soundness flags — any True is a hard failure (never mask these).
    false_optimum: bool = False  # certified optimal but disagrees with oracle
    bound_crosses: bool = False  # dual bound on the wrong side of the oracle
    note: str = ""


def _decide_objective(
    status: SolveStatus,
    candidate: float | None,
    violation: float | None,
) -> tuple[float | None, bool]:
    """Decide what objective to report for a run, and whether it needed verifying.

    Returns ``(objective, incumbent_verified)``.

    A run that ends at the time limit can still carry a real incumbent: the discopt
    pyomo bridge loads the solution whenever ``result.x`` is set, regardless of
    termination condition, so the point is sitting in the model. Discarding it on
    the status alone under-reports the primal — measured on cstr, discopt exits
    ``time_limit`` having found 3.3700327 (violation 1.6e-06, inside this file's
    own ``_ORACLE_FEAS_TOL``) and the sweep recorded "no incumbent".

    Such a point is reported ONLY if it verifies feasible in the real pyomo model,
    by the same standard applied to an oracle incumbent. An unevaluable or
    incompletely-loaded solution returns ``None`` and stays unreported — never
    treated as OK. Note the incumbent is genuinely absent, not merely unreported,
    on syngas and batch_processing, where no point is loaded at all.

    ``incumbent_verified`` is what re-arms the soundness checks: ``is_feasible`` is
    OPTIMAL|FEASIBLE only, so without it a reported time-limit objective would slip
    past the impossible-incumbent test (``CLAUDE.md`` §1).
    """
    if status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
        return candidate, False
    if candidate is not None and violation is not None and violation <= _ORACLE_FEAS_TOL:
        return candidate, True
    return None, False


def solve_model(
    spec: GDPModelSpec,
    method: str = "bigm",
    time_limit: float = 300.0,
    oracle: bool = True,
    max_variables: int | None = None,
) -> ModelRun:
    """Build, reformulate, and solve one GDPlib model with discopt.

    Returns a :class:`ModelRun`. Build/transform/solve failures are captured as an
    ``ERROR`` result rather than raised, so a sweep is robust to one bad model.
    """
    import discopt.pyomo  # noqa: F401  registers SolverFactory('discopt')
    import pyomo.environ as pyo
    from pyomo.core import TransformationFactory

    run_name = f"{spec.name}/{method}"

    def _err(msg: str, info: InstanceInfo | None = None) -> ModelRun:
        info = info or InstanceInfo(name=run_name, source="gdplib")
        info.name = run_name
        return ModelRun(
            name=run_name,
            info=info,
            discopt=SolveResult(instance=run_name, solver="discopt", status=SolveStatus.ERROR),
            is_linear=False,
            minimize=True,
            note=msg,
        )

    try:
        model = spec.builder()
    except Exception as exc:  # noqa: BLE001
        return _err(f"build failed: {type(exc).__name__}: {exc}")
    if model is None:
        return _err("builder returned None")

    try:
        TransformationFactory(f"gdp.{method}").apply_to(model)
    except Exception as exc:  # noqa: BLE001
        return _err(f"gdp.{method} transform failed: {type(exc).__name__}: {exc}")

    info, is_linear = _classify(model)
    info.name = run_name
    minimize = _is_minimize(model)
    if max_variables is not None and info.num_variables > max_variables:
        return ModelRun(
            name=run_name,
            info=info,
            discopt=SolveResult(instance=run_name, solver="discopt", status=SolveStatus.UNKNOWN),
            is_linear=is_linear,
            minimize=minimize,
            note=f"skipped: {info.num_variables} vars > max_variables={max_variables}",
        )

    t0 = time.time()
    try:
        res = pyo.SolverFactory("discopt").solve(model, options={"time_limit": time_limit})
    except Exception as exc:  # noqa: BLE001
        run = _err(f"discopt.solve failed: {type(exc).__name__}: {exc}", info)
        run.discopt.wall_time = time.time() - t0
        return run
    wall = time.time() - t0

    tc = str(res.solver.termination_condition).lower()
    status = _STATUS_MAP.get(tc, SolveStatus.UNKNOWN)
    candidate = _objective_value(model)
    violation = (
        None
        if status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)
        else _max_constraint_violation(model)
    )
    objective, incumbent_verified = _decide_objective(status, candidate, violation)
    bound = _problem_bound(res, minimize)
    nodes = _node_count(res)

    discopt_result = SolveResult(
        instance=run_name,
        solver="discopt",
        status=status,
        objective=objective,
        bound=bound,
        wall_time=wall,
        node_count=nodes,
    )

    run = ModelRun(
        name=run_name,
        info=info,
        discopt=discopt_result,
        is_linear=is_linear,
        minimize=minimize,
        incumbent_verified=incumbent_verified,
    )
    if oracle:
        _attach_oracle(run, spec, method, time_limit)
    _assess(run)
    return run


def _problem_bound(res, minimize: bool) -> float | None:
    prob = res.problem
    # Pyomo stores the dual bound in the field opposite the incumbent's.
    field_name = "upper_bound" if minimize else "lower_bound"
    try:
        val = getattr(prob[0] if hasattr(prob, "__getitem__") else prob, field_name, None)
    except Exception:
        val = None
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _node_count(res) -> int:
    try:
        return int(res.solver.statistics.branch_and_bound.number_of_nodes)
    except Exception:
        return 0


# Feasibility tolerance for accepting an oracle's incumbent as a real feasible point
# (scale-normalized per constraint — see :func:`_max_constraint_violation`).
_ORACLE_FEAS_TOL = 1e-5

# Integrality tolerance for an oracle's incumbent: the project integrality tolerance
# (1e-5, CLAUDE.md "Numerical tolerances"). It only decides which points are worth
# judging. The point actually judged is the *exactly integral* one -- each integer
# variable rounded -- because no tolerance is safe against a big-M row on its own:
# GAMS/SCIP returned ``67.3236`` on ``pyomo.eight_process`` (optimum ``68.00974``)
# with binaries at ``9.5e-7`` / ``0.999999``, each inside any reasonable tolerance,
# whose ``M * y`` slack relaxed the disjunct it was supposed to switch off.
_ORACLE_INT_TOL = 1e-5


def _referenced_variables(pyomo_model) -> list:
    """The variables an active constraint or objective references, once each."""
    import pyomo.environ as pyo
    from pyomo.core.expr.visitor import identify_variables

    referenced: dict[int, object] = {}
    for comp in pyomo_model.component_data_objects(pyo.Constraint, active=True):
        for v in identify_variables(comp.body, include_fixed=True):
            referenced.setdefault(id(v), v)
    for comp in pyomo_model.component_data_objects(pyo.Objective, active=True):
        for v in identify_variables(comp.expr, include_fixed=True):
            referenced.setdefault(id(v), v)
    return list(referenced.values())


def _snap_integers(pyomo_model) -> float | None:
    """Round every referenced integer/binary variable of the loaded point to the
    nearest integer **in place**; return the largest distance moved.

    ``None`` when a referenced integer variable is unset (solution not loaded). The
    oracle paths call this before :func:`_max_constraint_violation` and
    :func:`_objective_value`, so the value they certify is the objective *of the
    integral point they verified*, not of the fractional one the solver returned.
    A distance above :data:`_ORACLE_INT_TOL` means the point was not mixed-integer
    and the caller must reject it, rounded or not.
    """
    worst = 0.0
    for v in _referenced_variables(pyomo_model):
        if not v.is_integer():
            continue
        if v.value is None:
            return None
        val = float(v.value)
        snapped = float(round(val))
        worst = max(worst, abs(val - snapped))
        v.set_value(snapped, skip_validation=True)
    return worst


def _max_constraint_violation(pyomo_model) -> float | None:
    """Largest **scale-normalized** violation of the solution currently loaded in
    *pyomo_model*, or ``None`` if any active constraint/variable cannot be evaluated.

    This is the core of oracle hardening (#823). A claimed optimum *below* the true
    optimum is exactly a claimed feasible point that is **not** feasible (no feasible
    point beats the minimum), so an oracle value is trustworthy only if its incumbent,
    evaluated in the real pyomo model, actually satisfies every constraint. The
    pyscipopt-``.nl`` path could not be verified this way — its solution does not map
    back through the ``gdp.bigm`` indicator-variable aliases — which is precisely how
    it certified a below-true cstr optimum unchecked. A ``None`` return (unevaluable /
    incompletely loaded solution) is treated as *not certified feasible*, never as OK.

    Per-constraint violation is normalized by ``1 + |body|`` so a large-magnitude
    balance (e.g. water_network ~3e5) is judged on relative, not absolute, slack.

    Only variables an active constraint or objective references are checked
    (#1775 G0.5). A variable nothing references cannot affect feasibility, and the
    ``appsi`` interfaces do not load one: GDPlib's ``med_term_purchasing`` has 667,
    all unreferenced, which made a HiGHS-proven, verified-feasible optimum read as
    "solution not fully loaded". A *referenced* unset variable still returns ``None``.

    Integrality is part of feasibility (#1802 S2). An integer/binary variable more
    than :data:`_ORACLE_INT_TOL` from an integer makes the point infeasible
    (``inf``); within it, the rows are evaluated at the **rounded** point, so a
    near-integral binary cannot buy slack through a big-M row (see
    :data:`_ORACLE_INT_TOL` for the instance that did). The rounding is undone
    before returning; :func:`_snap_integers` is the in-place version.
    """
    import math

    referenced = _referenced_variables(pyomo_model)
    restore: list[tuple[object, float]] = []
    for v in referenced:
        if not v.is_integer() or v.value is None:
            continue
        val = float(v.value)
        if abs(val - round(val)) > _ORACLE_INT_TOL:
            return math.inf
        restore.append((v, v.value))
        v.set_value(float(round(val)), skip_validation=True)
    try:
        return _max_violation_at_loaded_point(pyomo_model, referenced)
    finally:
        for v, val in restore:
            v.set_value(val, skip_validation=True)


def _max_violation_at_loaded_point(pyomo_model, referenced: list) -> float | None:
    """The row and variable-bound part of :func:`_max_constraint_violation`."""
    import pyomo.environ as pyo

    worst = 0.0
    for c in pyomo_model.component_data_objects(pyo.Constraint, active=True):
        try:
            body = float(pyo.value(c.body))
        except Exception:
            return None  # a constraint we cannot evaluate -> feasibility unproven
        scale = 1.0 + abs(body)
        if c.has_lb():
            lo = float(pyo.value(c.lower))
            worst = max(worst, (lo - body) / scale)
        if c.has_ub():
            hi = float(pyo.value(c.upper))
            worst = max(worst, (body - hi) / scale)
    for v in referenced:
        val = v.value
        if val is None:
            return None  # an unset variable -> solution not fully loaded
        if v.lb is not None:
            worst = max(worst, (float(v.lb) - float(val)) / (1.0 + abs(float(v.lb))))
        if v.ub is not None:
            worst = max(worst, (float(val) - float(v.ub)) / (1.0 + abs(float(v.ub))))
    return worst


def _attach_oracle(run: ModelRun, spec: GDPModelSpec, method: str, time_limit: float) -> None:
    """Populate ``run.oracle_objective`` / ``oracle_source`` with a **verified** oracle.

    Every oracle value is trusted only if the solver *proved* optimality **and** the
    returned incumbent is confirmed feasible in the real pyomo model
    (:func:`_max_constraint_violation` ≤ tol). This closes the #823 hole where an
    unverifiable oracle seeded a below-true optimum that could mask a discopt false
    primal. Priority, most-trusted first:

    1. **HiGHS** on a *linear* reformulation — exact, independent, feasibility-verified.
    2. **Global via GAMS** (SCIP and BARON) — a solution-loadable path, so the
       incumbent is feasibility-verifiable (unlike pyscipopt-``.nl``). When both
       solvers return a verified proven optimum they must **agree**; a disagreement is
       refused loudly (never trust an inconsistent oracle). One verified solver alone
       is sufficient for soundness (its incumbent is a real feasible point).
    3. **Verified reference optimum** (:func:`reference_optima`) — the BARON-confirmed
       table, used offline / when GAMS is absent (e.g. CI).

    The pyscipopt-``.nl`` path is **deliberately no longer trusted** here: its solution
    cannot be mapped back for verification, and it demonstrably certified a false cstr
    optimum. `scripts/reeval_gdplib.py` still shows it as a raw diagnostic column.
    """
    if run.is_linear:
        obj = _solve_with_highs(spec, method, time_limit)
        if obj is not None:
            run.oracle_objective = obj
            run.oracle_source = "highs"
            return

    why: list[str] = []
    scip = _solve_with_gams(spec, method, time_limit, "scip", why=why)
    baron = _solve_with_gams(spec, method, time_limit, "baron", why=why)
    if why:
        run.note = "; ".join([run.note, *why] if run.note else why)
    if scip is not None and baron is not None:
        tol = 1e-4 + 1e-3 * max(abs(scip), abs(baron))
        if abs(scip - baron) > tol:
            run.note = (
                f"{run.note} ORACLE INCONSISTENCY: GAMS/SCIP={scip:.8g} vs "
                f"GAMS/BARON={baron:.8g} — refusing to trust either as the certified optimum"
            ).strip()
            return  # two proven+verified solvers must agree; if not, trust neither
        run.oracle_objective = scip
        run.oracle_source = "scip+baron"
        return
    if scip is not None:
        run.oracle_objective, run.oracle_source = scip, "scip-gams"
        return
    if baron is not None:
        run.oracle_objective, run.oracle_source = baron, "baron-gams"
        return

    ref = reference_optima().get(spec.name)
    if ref is None:
        # #1775 G0.5: a parity-panel instance certified offline (HiGHS on both
        # reformulations) anchors the run the same way when no live oracle ran.
        rec = panel_oracle().get(spec.name)
        ref = rec.value if rec is not None else None
    if ref is not None:
        run.oracle_objective = ref
        run.oracle_source = "reference"


def _solve_with_highs(spec: GDPModelSpec, method: str, time_limit: float) -> float | None:
    """Solve the (linear) reformulation with HiGHS as an independent oracle.

    Returns HiGHS's objective **only if HiGHS proved global optimality** within
    ``time_limit`` (termination ``optimal``) **and** the returned incumbent is verified
    feasible in the pyomo model; otherwise ``None``. An unconverged MILP yields only a
    suboptimal incumbent (would flag discopt's correct optimum as impossible, #823
    finding #1), and an infeasible/incompletely-loaded incumbent must never seed the
    oracle (the general #823 hardening).
    """
    import pyomo.environ as pyo
    from pyomo.core import TransformationFactory
    from pyomo.opt import TerminationCondition

    try:
        solver = pyo.SolverFactory("appsi_highs")
        if not solver.available(exception_flag=False):
            return None
        m = spec.builder()
        TransformationFactory(f"gdp.{method}").apply_to(m)
        # Bound HiGHS's runtime so a hard MILP cannot hang the sweep, and demand a
        # proven optimum (default rel/abs MIP gaps ≈ 0) before trusting the value.
        solver.config.time_limit = float(time_limit)
        results = solver.solve(m)
        if results.solver.termination_condition != TerminationCondition.optimal:
            return None
        dev = _snap_integers(m)
        if dev is None or dev > _ORACLE_INT_TOL:
            return None  # not a mixed-integer point -> not a usable oracle
        viol = _max_constraint_violation(m)
        if viol is None or viol > _ORACLE_FEAS_TOL:
            return None  # incumbent not confirmed feasible -> not a usable oracle
        return _objective_value(m)
    except Exception:  # noqa: BLE001
        return None


def _solve_with_gams(
    spec: GDPModelSpec,
    method: str,
    time_limit: float,
    solver: str,
    *,
    why: list[str] | None = None,
) -> float | None:
    """Solve the reformulation with a global solver via **GAMS** as a verified oracle.

    ``solver`` is a GAMS subsolver name (``"scip"`` / ``"baron"``). Returns its
    objective **only if** (a) the optimality gap genuinely closed (``lb == ub`` — GAMS
    can tag a time-limit incumbent ``optimal``, so we check the bounds, not the label)
    **and** (b) the incumbent GAMS loaded back into the pyomo model, with its integer
    variables rounded (:func:`_snap_integers`), is verified feasible
    (:func:`_max_constraint_violation` ≤ tol). The value returned is the objective of
    that rounded, verified point. Otherwise ``None``.

    GAMS round-trips the solution into the pyomo model — including the ``gdp.bigm``
    indicator aliases the pyscipopt-``.nl`` path mangled — so, unlike that path, the
    returned optimum is feasibility-checkable. This is the mechanism that would have
    rejected the false cstr optimum (#823).

    Every ``None`` has a reason, appended to *why* when given (#1802 S2): GAMS
    unavailable, a BARON run on a demo license (it caps at 10 variables / 10
    constraints, and its ``.lst`` says so), a failure writing or running the GAMS
    model (Pyomo's GAMS writer raises ``OverflowError`` on the hull form of
    ``pyomo.eight_process``), an open gap, a fractional or infeasible incumbent. The
    old blanket ``except Exception: return None`` hid every one of them, and a
    defect in this function's own code with them; only the GAMS call is guarded now.
    """
    import tempfile
    from pathlib import Path

    import pyomo.environ as pyo
    from pyomo.common.errors import ApplicationError
    from pyomo.core import TransformationFactory

    def _decline(reason: str) -> None:
        if why is not None:
            why.append(f"gams/{solver}: {reason}")

    gams = pyo.SolverFactory("gams")
    if not gams.available(exception_flag=False):
        _decline("GAMS not available")
        return None
    m = spec.builder()
    TransformationFactory(f"gdp.{method}").apply_to(m)
    with tempfile.TemporaryDirectory(prefix="gdp_oracle_") as tmp:
        try:
            # optcr defaults to 0.1 in GAMS — force a full gap closure, capped by reslim.
            res = gams.solve(
                m,
                solver=solver,
                add_options=[f"option reslim={float(time_limit)}; option optcr=1e-9;"],
                tee=False,
                keepfiles=True,
                tmpdir=tmp,
            )
        except (ApplicationError, OSError, OverflowError, RuntimeError, ValueError) as exc:
            _decline(f"GAMS solve failed: {type(exc).__name__}: {exc}")
            return None
        lst = Path(tmp) / "output.lst"
        listing = lst.read_text(errors="replace") if lst.exists() else ""
    if solver == "baron" and "demo license" in listing.lower():
        _decline("BARON ran on a demo license (10 vars / 10 cons) -- nothing certified")
        return None
    prob = res.problem[0] if hasattr(res.problem, "__getitem__") else res.problem
    lb, ub = getattr(prob, "lower_bound", None), getattr(prob, "upper_bound", None)
    if lb is None or ub is None:
        _decline(f"no bounds reported (lb={lb}, ub={ub})")
        return None
    lb, ub = float(lb), float(ub)
    if not (abs(lb - ub) <= 1e-6 + 1e-6 * abs(ub)):
        _decline(f"gap not closed (lb={lb!r}, ub={ub!r})")
        return None  # incumbent only, not a certified optimum
    dev = _snap_integers(m)
    if dev is None or dev > _ORACLE_INT_TOL:
        _decline(f"incumbent not integral (max integrality deviation {dev!r})")
        return None
    viol = _max_constraint_violation(m)
    if viol is None or viol > _ORACLE_FEAS_TOL:
        # A proven "optimum" whose point isn't feasible -> reject (#823). On
        # eight_process this is the rounded big-M point: feasible only with its
        # binaries at 9.5e-7 / 0.999999.
        _decline(f"incumbent infeasible at its integral rounding (violation {viol!r})")
        return None
    return _objective_value(m)


def _assess(run: ModelRun) -> None:
    """Set the soundness flags on *run* by comparing discopt to its oracle.

    Three independent checks (all must stay clean — see ``CLAUDE.md`` §1):

    * **impossible incumbent** — *any* feasible incumbent (even one merely reported
      ``FEASIBLE``, not ``OPTIMAL``) whose objective is *strictly better* than the
      true optimum (lower for min, higher for max). No feasible point can beat the
      optimum, so this means the incumbent violates a constraint — a false primal
      (issue #815). This is the most dangerous flag and does not require an
      ``OPTIMAL`` claim.
    * **false optimum** — discopt reports ``OPTIMAL`` but its certified objective
      disagrees with the oracle beyond tolerance (in either direction).
    * **bound crossing** — discopt's dual bound sits on the far/wrong side of the
      oracle optimum (would fathom the true optimum).
    """
    r = run.discopt
    opt = run.oracle_objective
    if opt is None:
        return
    abs_tol, rel_tol = 1e-4, 1e-3
    tol = abs_tol + rel_tol * abs(opt)

    # ``is_feasible`` is OPTIMAL|FEASIBLE only, so a verified incumbent reported
    # from a ``time_limit`` run would otherwise skip the impossible-incumbent test
    # below — the single most dangerous check here. Reporting that incumbent
    # without widening this gate would have created an unchecked path.
    if (r.is_feasible or run.incumbent_verified) and r.objective is not None:
        beats_oracle = (run.minimize and r.objective < opt - tol) or (
            not run.minimize and r.objective > opt + tol
        )
        disagrees = abs(r.objective - opt) > tol
        if beats_oracle:
            # Impossible: no feasible point beats the optimum -> false primal.
            run.false_optimum = True
            run.note = (
                f"IMPOSSIBLE INCUMBENT ({r.status.value}): discopt={r.objective:.8g} "
                f"beats oracle={opt:.8g} [{run.oracle_source}] — incumbent is infeasible"
            )
        elif r.is_solved and disagrees:
            # Claimed optimal but converged to a suboptimal (worse) objective.
            run.false_optimum = True
            run.note = (
                f"INCORRECT (worse-than-oracle, claimed optimal): "
                f"discopt={r.objective:.8g} oracle={opt:.8g} [{run.oracle_source}]"
            )

    if dual_bound_crosses_optimum(r.bound, opt, run.minimize, abs_tol, rel_tol):
        run.bound_crosses = True
        run.note = (f"{run.note} | BOUND CROSSES oracle {opt:.8g}").strip(" |")


def run_suite(config: GDPLibSuiteConfig) -> tuple[BenchmarkResults, list[ModelRun]]:
    """Run a full GDPlib sweep per *config*. Returns ``(BenchmarkResults, runs)``."""
    if not is_available():
        raise RuntimeError(
            "GDPlib benchmark requires pyomo + gdplib: "
            "pip install 'discopt-benchmarks[gdplib]' and install gdplib from source "
            "(the PyPI wheel omits model data files)."
        )
    specs = discover_models(include=config.include, exclude=config.exclude)
    results = BenchmarkResults(suite=config.name, timestamp=datetime.now().isoformat())
    runs: list[ModelRun] = []
    for spec in specs:
        for method in config.methods:
            run = solve_model(
                spec,
                method=method,
                time_limit=config.time_limit_seconds,
                oracle=config.oracle,
                max_variables=config.max_variables,
            )
            results.instance_info[run.name] = run.info
            results.add_result(run.discopt)
            runs.append(run)
            _print_run(run)
    _print_summary(runs, oracle_enabled=config.oracle)
    return results, runs


def sweep_is_vacuous(runs: list[ModelRun], oracle_enabled: bool) -> bool:
    """True when a sweep that was *meant* to verify something verified nothing.

    ``CLAUDE.md`` §6: every soundness check here is gated on ``oracle_objective is
    not None`` (see :func:`_assess`, which returns early without it). If no run
    acquires an oracle -- every model errored on build, every model was skipped by
    ``--max-variables``, or no oracle backend is installed -- then zero comparisons
    execute, ``violations`` is trivially 0, and the sweep reports "no soundness
    violations" while having verified nothing. That is a no-op wearing a pass, so
    the caller must fail on it instead.

    ``--no-oracle`` is the one honest way to check nothing, so it is exempt.
    """
    if not oracle_enabled:
        return False
    return not any(r.oracle_objective is not None for r in runs)


def _print_run(run: ModelRun) -> None:
    r = run.discopt
    obj = f"{r.objective:.6g}" if r.objective is not None else "—"
    flag = ""
    if run.false_optimum or run.bound_crosses:
        flag = "  ✗ SOUNDNESS"
    elif run.oracle_source and r.is_solved:
        flag = f"  ✓ vs {run.oracle_source} ({run.oracle_objective:.6g})"
    elif run.oracle_source and r.is_feasible:
        # Feasible but not proven optimal: show the certified optimum + the gap
        # to it, so a loose incumbent is visible (and its soundness is confirmed).
        flag = f"  ~ {run.oracle_source} opt={run.oracle_objective:.6g}"
    line = (
        f"  {run.name:42s} {r.status.value:11s} obj={obj:>12s} "
        f"nodes={r.node_count:>7d} {r.wall_time:6.1f}s{flag}"
    )
    print(line)
    if run.note:
        print(f"      · {run.note}")


def _print_summary(runs: list[ModelRun], oracle_enabled: bool = True) -> None:
    n = len(runs)
    solved = sum(1 for r in runs if r.discopt.is_solved)
    feasible = sum(1 for r in runs if r.discopt.is_feasible)
    errored = sum(1 for r in runs if r.discopt.status == SolveStatus.ERROR)
    checked = sum(1 for r in runs if r.oracle_objective is not None)
    incorrect = sum(1 for r in runs if r.false_optimum)
    bound_bad = sum(1 for r in runs if r.bound_crosses)
    print("\n" + "=" * 68)
    print(f"GDPlib sweep: {n} runs | solved={solved} feasible={feasible} error={errored}")
    print(f"oracle-checked={checked} | INCORRECT={incorrect} | bound-crossings={bound_bad}")
    if incorrect or bound_bad:
        print("  ✗ SOUNDNESS VIOLATIONS — see flagged runs above (incorrect_count must be 0)")
    elif sweep_is_vacuous(runs, oracle_enabled):
        # Never let "verified nothing" print as "verified clean" (CLAUDE.md §6).
        print("  ✗ VACUOUS SWEEP — oracle-checked=0: zero comparisons executed, so")
        print("    'no violations' means 'nothing was checked', not 'nothing is wrong'.")
    elif not oracle_enabled:
        print("  — oracle disabled (--no-oracle): no soundness claim is made")
    else:
        print(f"  ✓ no soundness violations across {checked} oracle-checked runs")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark discopt on the GDPlib corpus.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="explicit model names (default: all discovered). e.g. jobshop cstr",
    )
    parser.add_argument("--exclude", nargs="*", default=None, help="model names to skip")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default=None,
        help="named model set: gdplib_small (the 12 models under 500 vars) or "
        "gdp_parity (#1775: GDPlib PR58 + Pyomo's GDPopt test models). "
        "Mutually exclusive with --models.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bigm"],
        choices=["bigm", "hull"],
        help="GDP reformulation(s) to apply (default: bigm)",
    )
    parser.add_argument("--time-limit", type=float, default=300.0, help="per-solve seconds")
    parser.add_argument("--max-variables", type=int, default=None, help="skip larger models")
    parser.add_argument("--no-oracle", action="store_true", help="skip HiGHS/reference checks")
    parser.add_argument("--list", action="store_true", help="list discovered models and exit")
    parser.add_argument("--output", default=None, help="write BenchmarkResults JSON to this path")
    args = parser.parse_args(argv)

    if args.suite is not None:
        if args.models:
            parser.error("--suite and --models are mutually exclusive")
        args.models = list(SUITES[args.suite])

    if not is_available():
        print(
            "GDPlib benchmark unavailable: install pyomo + gdplib "
            "(gdplib from source — the PyPI wheel omits data files)."
        )
        return 2

    if args.suite is not None:
        # `include` filtering in discover_models() drops a requested-but-absent
        # model silently. For a preset pinned by name that is the §6 vacuity trap:
        # a panel that ran 8 of 12 models and reported clean. Refuse loudly.
        found = {s.name for s in discover_models(include=args.models, exclude=args.exclude)}
        missing = [m for m in args.models if m not in found]
        if missing:
            print(
                f"FAIL: suite '{args.suite}' names {len(missing)} model(s) this gdplib "
                f"install does not provide: {', '.join(missing)}. A shrunken suite would "
                "report clean while measuring less than it claims — install gdplib from "
                "source, or run the models you do have via --models.",
                file=sys.stderr,
            )
            return 4

    if args.list:
        specs = discover_models(include=args.models, exclude=args.exclude)
        print(f"{len(specs)} runnable GDPlib models:")
        for s in specs:
            print(f"  {s.name}")
        return 0

    config = GDPLibSuiteConfig(
        methods=tuple(args.methods),
        time_limit_seconds=args.time_limit,
        max_variables=args.max_variables,
        include=args.models,
        exclude=args.exclude or [],
        oracle=not args.no_oracle,
    )
    results, runs = run_suite(config)
    if args.output:
        from pathlib import Path

        results.save(Path(args.output))
        print(f"wrote results to {args.output}")
    # Nonzero exit on any soundness violation so CI can gate on it.
    violations = sum(1 for r in runs if r.false_optimum or r.bound_crosses)
    if violations:
        return 1
    if sweep_is_vacuous(runs, config.oracle):
        # §6: a sweep that executed zero comparisons must not exit green.
        print(
            "FAIL: oracle-checked=0 — this sweep verified nothing. Install an oracle "
            "(highspy for the linear subset, GAMS/SCIP for the nonlinear one), widen "
            "--max-variables, or pass --no-oracle to say so deliberately.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
