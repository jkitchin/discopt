"""Native-discopt GDPlib models (issue #823).

Companion to :mod:`benchmarks.gdplib_runner`. That runner drives discopt through
the **Pyomo** ``gdp.bigm`` / ``gdp.hull`` bridge — so the disjunctions are lowered
to a MI(N)LP by *Pyomo* before discopt ever sees them. This module instead rebuilds
a curated subset of the same GDPlib problems **directly in discopt's native
modeling API** (:meth:`Model.either_or` / :meth:`Model.add_disjunction`), so
discopt's *own* disjunction machinery — its in-house big-M/hull lowering and the
integrality-aware FBBT that branches on the selector binaries — is what gets
exercised. Two independent lowerings of the same math are a cross-check on both.

Each native builder is verified against the **same SCIP-certified optimum** as the
Pyomo-bridged model (:func:`benchmarks.gdplib_runner.reference_optima`); a native
model that reaches a *different* optimum is a porting bug and fails its test.
Unlike the runner, these builders need **only discopt** — not pyomo or gdplib —
because the model is transcribed, not imported.

Porting methodology (so the set can grow honestly): read the gdplib Pyomo source,
transcribe variables/params/constraints verbatim, and encode each ``Disjunction``
with the native disjunctive API. A GDP's optimum is reformulation-independent, so
an equivalent native encoding of the *feasible set* (e.g. collapsing a per-unit
``exists``/``not`` disjunction plus its ``exactly(1)`` selector into one k-way
``either_or`` over the resulting discrete value) is faithful as long as it reaches
the certified optimum — which every builder here is tested to do.

Models deliberately **not yet ported** (candidates for extension), with reason:
``cstr`` (944-LOC reactor superstructure with recycle — high transcription risk),
``positioning`` (25×5 embedded data tables), ``syngas`` / ``water_network`` /
``methanol`` / ``modprodnet`` / ``batch_processing`` / ``gdp_col`` (large process
models). Port from source and add a certified-optimum test before listing them.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarks.gdplib_runner import _STATUS_MAP, ModelRun, _assess, reference_optima
from benchmarks.metrics import InstanceInfo, SolveResult, SolveStatus

if TYPE_CHECKING:
    from collections.abc import Callable


# ── Native model builders ────────────────────────────────────────────────────
#
# Each returns a discopt ``Model`` with the objective set. All are minimize.


def build_jobshop():
    """Jobshop scheduling (Raman & Grossmann 1994), the ``jobshop-small`` data.

    Minimize the makespan of 3 jobs over 3 stages under a zero-wait policy, with a
    two-way disjunction (I-before-K vs K-before-I) forbidding a clash wherever two
    jobs share a stage. Certified optimum 11.0 (also HiGHS-checkable — it is linear).
    """
    from discopt import Model

    # tau[job][stage], 0 = job skips that stage (jobshop-small.dat).
    tau = {"A": {1: 5, 2: 0, 3: 3}, "B": {1: 0, 2: 3, 3: 2}, "C": {1: 2, 2: 4, 3: 0}}
    jobs, stages = ["A", "B", "C"], [1, 2, 3]
    horizon = float(sum(tau[j][s] for j in jobs for s in stages))

    m = Model()
    ms = m.continuous("ms", lb=0, ub=horizon)
    t = {j: m.continuous(f"t_{j}", lb=0, ub=horizon) for j in jobs}
    m.minimize(ms)

    # Makespan dominates every job's completion (start + total duration).
    for j in jobs:
        m.subject_to(ms >= t[j] + sum(tau[j][s] for s in stages))

    # A clash is possible only where jobs I<K both use stage J. Prefix time before
    # J is start + earlier-stage durations; the chosen order must not overlap at J.
    for i_idx, i in enumerate(jobs):
        for k in jobs[i_idx + 1 :]:
            for j in stages:
                if tau[i][j] and tau[k][j]:
                    pre_i = t[i] + sum(tau[i][s] for s in stages if s < j)
                    pre_k = t[k] + sum(tau[k][s] for s in stages if s < j)
                    m.either_or(
                        [
                            [pre_i + tau[i][j] <= pre_k],  # I before K
                            [pre_k + tau[k][j] <= pre_i],  # K before I
                        ],
                        name=f"noclash_{i}_{k}_{j}",
                    )
    return m


def build_ex1_linan_2023():
    """Toy GDP of Liñán & Ricardez-Sandoval (2023): six-hump-camel over a grid.

    Minimize the (nonconvex) six-hump camel function while two xor disjunctions
    pin ``alpha`` and ``beta`` to grid points. A logical constraint forces the
    third ``alpha`` disjunct false, so ``alpha = 0.1`` is removed from its grid.
    Certified optimum -0.9996.
    """
    from discopt import Model

    m = Model()
    a = m.continuous("alpha", lb=-0.1, ub=0.4)
    b = m.continuous("beta", lb=-0.9, ub=-0.5)
    a2 = a * a
    a4 = a2 * a2
    a6 = a4 * a2
    b2 = b * b
    b4 = b2 * b2
    m.minimize(4 * a2 - 2.1 * a4 + (1.0 / 3.0) * a6 + a * b - 4 * b2 + 4 * b4)

    # alpha grid = {-0.1, 0.0, 0.1, 0.2, 0.3}; the infeasR logical constraint fixes
    # Y1[3] = False, dropping the j=3 value (0.1). beta grid = {-0.9,...,-0.5}.
    alpha_grid = [-0.1 + 0.1 * (j - 1) for j in (1, 2, 4, 5)]  # j=3 (0.1) excluded
    beta_grid = [-0.9 + 0.1 * (j - 1) for j in (1, 2, 3, 4, 5)]
    m.either_or([[a == v] for v in alpha_grid], name="alpha_grid")
    m.either_or([[b == v] for v in beta_grid], name="beta_grid")
    return m


def build_small_batch():
    """Small batch-process design (Kocis & Grossmann 1988).

    Size 3 batch stages (mixer/reactor/centrifuge) making 2 products, choosing the
    number of parallel units per stage to minimize investment cost. The original
    per-(unit,stage) exists/not disjunction plus an ``exactly(1)`` selector fixes
    ``n[j] = log(k)`` for a single chosen ``k`` — encoded here as one 3-way
    disjunction per stage over ``n[j] ∈ {log 1, log 2, log 3}`` (same feasible set).
    Certified optimum 167427.65.
    """
    from discopt import Model
    from discopt.modeling import exp

    products, stages, units = ["a", "b"], ["mixer", "reactor", "centrifuge"], [1, 2, 3]
    horizon, vlow, vupp = 6000.0, 250.0, 2500.0
    q = {"a": 200000.0, "b": 150000.0}
    alpha = {"mixer": 250.0, "reactor": 500.0, "centrifuge": 340.0}
    beta = 0.6
    s = {
        ("a", "mixer"): 2, ("a", "reactor"): 3, ("a", "centrifuge"): 4,
        ("b", "mixer"): 4, ("b", "reactor"): 6, ("b", "centrifuge"): 3,
    }
    t = {
        ("a", "mixer"): 8, ("a", "reactor"): 20, ("a", "centrifuge"): 4,
        ("b", "mixer"): 10, ("b", "reactor"): 12, ("b", "centrifuge"): 3,
    }

    m = Model()
    v = {j: m.continuous(f"v_{j}", lb=math.log(vlow), ub=math.log(vupp)) for j in stages}
    n = {j: m.continuous(f"n_{j}", lb=0, ub=math.log(len(units))) for j in stages}
    b = {}
    tl = {}
    for i in products:
        ub_b = min(math.log(vupp / s[i, j]) for j in stages)
        b[i] = m.continuous(f"b_{i}", lb=0, ub=ub_b)
        tl[i] = m.continuous(f"tl_{i}", lb=0, ub=math.log(horizon / q[i]) + ub_b)

    for i in products:
        for j in stages:
            m.subject_to(v[j] >= math.log(s[i, j]) + b[i])  # volume requirement
            m.subject_to(n[j] + tl[i] >= math.log(t[i, j]))  # cycle time
    m.subject_to(sum(q[i] * exp(tl[i] - b[i]) for i in products) <= horizon)  # horizon

    # Choose parallel-unit count per stage: n[j] = log(k) for exactly one k.
    coeffs = [math.log(k) for k in units]
    for j in stages:
        m.either_or([[n[j] == c] for c in coeffs], name=f"units_{j}")

    m.minimize(sum(alpha[j] * exp(n[j] + beta * v[j]) for j in stages))
    return m


NATIVE_BUILDERS: dict[str, Callable[[], object]] = {
    "jobshop": build_jobshop,
    "ex1_linan_2023": build_ex1_linan_2023,
    "small_batch": build_small_batch,
}


# ── Solve + soundness cross-check ────────────────────────────────────────────


@dataclass
class NativeRun:
    """One native-discopt solve plus its cross-check against the certified optimum."""

    name: str
    discopt: SolveResult
    oracle_objective: float | None
    oracle_source: str  # "reference" (SCIP-certified) or "none"
    false_optimum: bool = False
    bound_crosses: bool = False
    note: str = ""


def solve_native(name: str, time_limit: float = 300.0) -> NativeRun:
    """Build the native model *name*, solve it, and assess soundness.

    The oracle is the SCIP-certified :func:`reference_optima` value (a GDP optimum
    is reformulation-independent, so the same value anchors the native encoding).
    Reuses :func:`benchmarks.gdplib_runner._assess` so the native and Pyomo-bridge
    paths share **one** soundness implementation (impossible-incumbent / false-
    optimum / bound-crossing — all must stay clean, ``CLAUDE.md`` §1).
    """
    if name not in NATIVE_BUILDERS:
        raise KeyError(f"no native builder for {name!r}; have {sorted(NATIVE_BUILDERS)}")

    t0 = time.time()
    try:
        model = NATIVE_BUILDERS[name]()
        res = model.solve(time_limit=time_limit)
    except Exception as exc:  # noqa: BLE001
        return NativeRun(
            name=name,
            discopt=SolveResult(instance=name, solver="discopt-native", status=SolveStatus.ERROR),
            oracle_objective=None,
            oracle_source="none",
            note=f"native solve failed: {type(exc).__name__}: {exc}",
        )
    wall = time.time() - t0

    status = _STATUS_MAP.get(str(res.status).lower(), SolveStatus.UNKNOWN)
    objective = res.objective if status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE) else None
    discopt_result = SolveResult(
        instance=name,
        solver="discopt-native",
        status=status,
        objective=objective,
        bound=getattr(res, "bound", None),
        wall_time=wall,
        node_count=int(getattr(res, "node_count", 0) or 0),
    )

    ref = reference_optima().get(name)
    run = ModelRun(
        name=name,
        info=InstanceInfo(name=name, source="gdplib-native"),
        discopt=discopt_result,
        is_linear=False,
        minimize=True,  # every native model here is a minimize
        oracle_objective=ref,
        oracle_source="reference" if ref is not None else None,
    )
    _assess(run)  # shared soundness gate
    return NativeRun(
        name=name,
        discopt=discopt_result,
        oracle_objective=ref,
        oracle_source="reference" if ref is not None else "none",
        false_optimum=run.false_optimum,
        bound_crosses=run.bound_crosses,
        note=run.note,
    )


def run_native_suite(names: list[str] | None = None, time_limit: float = 300.0) -> list[NativeRun]:
    """Solve every native model (or the given subset) and print a summary."""
    names = names or sorted(NATIVE_BUILDERS)
    runs: list[NativeRun] = []
    for name in names:
        run = solve_native(name, time_limit=time_limit)
        runs.append(run)
        _print_run(run)
    _print_summary(runs)
    return runs


def _print_run(run: NativeRun) -> None:
    r = run.discopt
    obj = f"{r.objective:.6g}" if r.objective is not None else "—"
    if run.false_optimum or run.bound_crosses:
        flag = "  ✗ SOUNDNESS"
    elif run.oracle_objective is not None and r.is_solved:
        flag = f"  ✓ vs {run.oracle_source} ({run.oracle_objective:.6g})"
    elif run.oracle_objective is not None and r.is_feasible:
        flag = f"  ~ {run.oracle_source} opt={run.oracle_objective:.6g}"
    else:
        flag = ""
    print(
        f"  {run.name:20s} {r.status.value:11s} obj={obj:>12s} "
        f"nodes={r.node_count:>7d} {r.wall_time:6.1f}s{flag}"
    )
    if run.note:
        print(f"      · {run.note}")


def _print_summary(runs: list[NativeRun]) -> None:
    n = len(runs)
    solved = sum(1 for r in runs if r.discopt.is_solved)
    incorrect = sum(1 for r in runs if r.false_optimum)
    bound_bad = sum(1 for r in runs if r.bound_crosses)
    print("\n" + "=" * 60)
    print(f"native GDPlib: {n} runs | solved={solved} | "
          f"INCORRECT={incorrect} | bound-crossings={bound_bad}")
    if incorrect or bound_bad:
        print("  ✗ SOUNDNESS VIOLATIONS — see flagged runs (incorrect_count must be 0)")
    else:
        print("  ✓ no soundness violations")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Solve native-discopt GDPlib models.")
    parser.add_argument("--models", nargs="*", default=None, help="subset (default: all)")
    parser.add_argument("--time-limit", type=float, default=300.0, help="per-solve seconds")
    parser.add_argument("--list", action="store_true", help="list native models and exit")
    args = parser.parse_args(argv)

    if args.list:
        print(f"{len(NATIVE_BUILDERS)} native GDPlib models:")
        for name in sorted(NATIVE_BUILDERS):
            print(f"  {name}")
        return 0

    runs = run_native_suite(names=args.models, time_limit=args.time_limit)
    violations = sum(1 for r in runs if r.false_optimum or r.bound_crosses)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
