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
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from benchmarks.metrics import (
    BenchmarkResults,
    InstanceInfo,
    SolveResult,
    SolveStatus,
    dual_bound_crosses_optimum,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# discopt status string -> benchmark SolveStatus.
_STATUS_MAP = {
    "optimal": SolveStatus.OPTIMAL,
    "feasible": SolveStatus.FEASIBLE,
    "infeasible": SolveStatus.INFEASIBLE,
    "unbounded": SolveStatus.UNBOUNDED,
    "time_limit": SolveStatus.TIME_LIMIT,
    "maxtimelimit": SolveStatus.TIME_LIMIT,
    "node_limit": SolveStatus.TIME_LIMIT,
    "maxiterations": SolveStatus.TIME_LIMIT,
    "error": SolveStatus.ERROR,
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
    # Soundness flags — any True is a hard failure (never mask these).
    false_optimum: bool = False  # certified optimal but disagrees with oracle
    bound_crosses: bool = False  # dual bound on the wrong side of the oracle
    note: str = ""


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
    objective = (
        _objective_value(model)
        if status
        in (
            SolveStatus.OPTIMAL,
            SolveStatus.FEASIBLE,
        )
        else None
    )
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
    """
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
    for v in pyomo_model.component_data_objects(pyo.Var, active=True):
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

    scip = _solve_with_gams(spec, method, time_limit, "scip")
    baron = _solve_with_gams(spec, method, time_limit, "baron")
    if scip is not None and baron is not None:
        tol = 1e-4 + 1e-3 * max(abs(scip), abs(baron))
        if abs(scip - baron) > tol:
            run.note = (
                f"ORACLE INCONSISTENCY: GAMS/SCIP={scip:.8g} vs GAMS/BARON={baron:.8g} "
                "— refusing to trust either as the certified optimum"
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
        viol = _max_constraint_violation(m)
        if viol is None or viol > _ORACLE_FEAS_TOL:
            return None  # incumbent not confirmed feasible -> not a usable oracle
        return _objective_value(m)
    except Exception:  # noqa: BLE001
        return None


def _solve_with_gams(
    spec: GDPModelSpec, method: str, time_limit: float, solver: str
) -> float | None:
    """Solve the reformulation with a global solver via **GAMS** as a verified oracle.

    ``solver`` is a GAMS subsolver name (``"scip"`` / ``"baron"``). Returns its
    objective **only if** (a) the optimality gap genuinely closed (``lb == ub`` — GAMS
    can tag a time-limit incumbent ``optimal``, so we check the bounds, not the label)
    **and** (b) the incumbent GAMS loaded back into the pyomo model is verified feasible
    (:func:`_max_constraint_violation` ≤ tol). Otherwise ``None``.

    GAMS round-trips the solution into the pyomo model — including the ``gdp.bigm``
    indicator aliases the pyscipopt-``.nl`` path mangled — so, unlike that path, the
    returned optimum is feasibility-checkable. This is the mechanism that would have
    rejected the false cstr optimum (#823). ``None`` when GAMS is unavailable.
    """
    import pyomo.environ as pyo
    from pyomo.core import TransformationFactory

    try:
        gams = pyo.SolverFactory("gams")
        if not gams.available(exception_flag=False):
            return None
        m = spec.builder()
        TransformationFactory(f"gdp.{method}").apply_to(m)
        # optcr defaults to 0.1 in GAMS — force a full gap closure, capped by reslim.
        res = gams.solve(
            m,
            solver=solver,
            add_options=[f"option reslim={float(time_limit)}; option optcr=1e-9;"],
            tee=False,
        )
        prob = res.problem[0] if hasattr(res.problem, "__getitem__") else res.problem
        lb, ub = getattr(prob, "lower_bound", None), getattr(prob, "upper_bound", None)
        if lb is None or ub is None:
            return None
        lb, ub = float(lb), float(ub)
        if abs(lb - ub) > 1e-6 + 1e-6 * abs(ub):
            return None  # gap not closed -> incumbent only, not a certified optimum
        viol = _max_constraint_violation(m)
        if viol is None or viol > _ORACLE_FEAS_TOL:
            return None  # proven "optimum" whose point isn't feasible -> reject (#823)
        return _objective_value(m)
    except Exception:  # noqa: BLE001
        return None


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

    if r.is_feasible and r.objective is not None:
        beats_oracle = (
            (run.minimize and r.objective < opt - tol)
            or (not run.minimize and r.objective > opt + tol)
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
    _print_summary(runs)
    return results, runs


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


def _print_summary(runs: list[ModelRun]) -> None:
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
    else:
        print("  ✓ no soundness violations among oracle-checked runs")
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

    if not is_available():
        print(
            "GDPlib benchmark unavailable: install pyomo + gdplib "
            "(gdplib from source — the PyPI wheel omits data files)."
        )
        return 2

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
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
