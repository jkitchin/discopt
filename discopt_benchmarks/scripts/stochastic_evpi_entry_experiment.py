#!/usr/bin/env python3
"""Entry experiment for the scenario-decomposed global B&B (A8 / SNGO-SNoGloDe).

**Hypothesis.** On two-stage stochastic nonconvex NLPs, the lower bound obtained by
*relaxing non-anticipativity* — Cao & Zavala (2019), ``docs/references.bib``
``CaoZavala2019`` — is tighter at the root than discopt's monolithic convex-relaxation
root bound, and stays tight as the scenario count grows. The relaxation gap is exactly
the **expected value of perfect information (EVPI)**, so the whole method's efficiency
rests on EVPI being moderate on the target class.

**Why this gates the implementation.** A scenario-decomposed spatial B&B branches only
on the first-stage variables and pays S global subproblem solves per node. That trade is
worth taking only if the decomposed root bound is *better* than what we already get for
one relaxation solve. Cao & Zavala report root EVPI of 36.1 / 14.3 / 1.4 / 9.5 % against
SCIP root gaps of 101.5 / 62.6 / >=10000 / 10.6 % on four instances; this script asks
whether the same ordering holds against *discopt's* root bound.

**Measurement.** Per family and scenario count S:

* ``LB_dec``  -- sum over scenarios of the globally-solved subproblem bound over the full
  first-stage box (the node lower bound of the proposed method at the root).
* ``alpha``   -- the method's root upper bound: fix the first stage at the
  probability-weighted mean of the subproblem minimisers and re-solve each scenario
  globally (Cao & Zavala 2.2, with the mean candidate of their 4).
* ``root_bound`` -- discopt's own root bound on the monolithic extensive form
  (``SolveResult.root_bound``), replicated to expose its spread.
* Gaps are reported against the best incumbent either arm found, so the two bounds are
  compared on one scale.

**Kill criterion.** If ``decomposed_root_gap >= discopt_root_gap`` on a majority of
(family, S) cells, the decomposition has no bound headroom here and the implementation
does not get scheduled. A win must also not decay as S grows.

**Measurement discipline** (CLAUDE.md "Measurement & instrumentation discipline"):

* Every cell increments an explicit comparison counter; the script **exits non-zero if
  zero comparisons executed** (rule 6) -- a sweep that silently measured nothing must
  not read as a pass.
* Solve failures are recorded with their traceback and excluded from the count; nothing
  is swallowed (rule 7).
* ``discopt.__file__`` is asserted to be the worktree under test (rule 8).
* Every solve passes ``deterministic=True``; the monolith root bound is replicated and
  its spread reported, and system load is printed before and after (rule 9) -- note
  ``solver.py``'s retraction that a node budget alone does NOT buy reproducibility.
* Per-cell progress is printed with ``flush=True`` (rule 10).

Usage::

    python -u -m discopt_benchmarks.scripts.stochastic_evpi_entry_experiment \\
        [--scenarios 4,8,16] [--families pid,estimation,pooling] \\
        [--time-limit 60] [--replicates 3] [--out reports/evpi_entry.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections.abc import Callable  # noqa: TC003  (used at runtime in a dataclass field)
from dataclasses import asdict, dataclass, field

import numpy as np
from discopt import Model

# --------------------------------------------------------------------------------------
# Instance families.
#
# Each family exposes:
#   first_stage_spec() -> list[(name, lb, ub)]        the shared here-and-now variables
#   scenario_data(n_scen)   -> list[(probability, data)]   the realisations
#   add_recourse(model, fs_vars, data, s) -> Expression   probability-weighted cost
#
# so that the SAME builder assembles both the monolithic extensive form and the n_scen
# standalone subproblems. That is the property the eventual implementation needs from
# ``discopt.stochastic`` (its ``recourse_builder`` closes over the outer first-stage
# variables today, so it cannot be retargeted to a fresh Model -- see the plan doc).
# --------------------------------------------------------------------------------------


@dataclass
class Family:
    name: str
    first_stage_spec: Callable[[int], list[tuple[str, float, float]]]
    scenario_data: Callable[[int], list[tuple[float, dict]]]
    add_recourse: Callable[[Model, dict, dict, int], object]
    note: str
    # here-and-now cost c(x). Folded into EVERY subproblem as p_s * c(x_s) so that
    # sum_s p_s * (c(x_s) + f_s) reproduces the monolith exactly when the copies agree
    # (the same probability-weighting trick ``stochastic/lshaped.py`` uses).
    first_stage_cost: Callable[[dict], object] | None = None


# ---- Family A: optimal PID controller tuning (Cao & Zavala 5.1) ------------------------
#
#   min_{Kp,Ki,Kd}  sum_s p_s * int e_s(t)^2 dt
#   s.t.  dx_s/dt = -tau_x,s x_s^2 + tau_u,s u_s + tau_d,s d_s      (implicit Euler)
#         e_s = x_s - xbar_s
#         u_s = Kp e_s + Ki I_s + Kd de_s/dt
#
# Nonconvex through x^2 and through the bilinear products of the first-stage gains with
# the recourse trajectory -- i.e. the first stage enters the recourse nonlinearly, which
# is the case that makes the recourse value function nonconvex.

_PID_STEPS = 8
_PID_HORIZON = 2.0


def _pid_first_stage() -> list[tuple[str, float, float]]:
    return [("Kp", 0.0, 5.0), ("Ki", 0.0, 5.0), ("Kd", 0.0, 2.0)]


def _pid_scenarios(n_scen: int) -> list[tuple[float, dict]]:
    rng = np.random.default_rng(20260921)
    out = []
    for _ in range(n_scen):
        out.append(
            (
                1.0 / n_scen,
                {
                    "xbar": float(rng.uniform(0.8, 1.6)),
                    "tau_x": float(rng.uniform(0.3, 1.2)),
                    "tau_u": float(rng.uniform(0.8, 1.6)),
                    "tau_d": float(rng.uniform(0.1, 0.5)),
                    "d": float(rng.uniform(-0.4, 0.4)),
                },
            )
        )
    return out


def _pid_recourse(model: Model, fs: dict, data: dict, s: int):
    dt = _PID_HORIZON / _PID_STEPS
    xbar, tx, tu, td, dist = (
        data["xbar"],
        data["tau_x"],
        data["tau_u"],
        data["tau_d"],
        data["d"],
    )
    kp, ki, kd = fs["Kp"], fs["Ki"], fs["Kd"]

    x_prev = model.continuous(f"x_{s}_0", lb=0.0, ub=3.0)
    model.subject_to(x_prev == 0.0)
    e_prev = None
    integral_prev = model.continuous(f"I_{s}_0", lb=-20.0, ub=20.0)
    model.subject_to(integral_prev == 0.0)

    cost = None
    for k in range(1, _PID_STEPS + 1):
        x_k = model.continuous(f"x_{s}_{k}", lb=0.0, ub=3.0)
        e_k = model.continuous(f"e_{s}_{k}", lb=-3.0, ub=3.0)
        integral_k = model.continuous(f"I_{s}_{k}", lb=-20.0, ub=20.0)
        u_k = model.continuous(f"u_{s}_{k}", lb=-30.0, ub=30.0)

        model.subject_to(e_k == x_k - xbar)
        model.subject_to(integral_k == integral_prev + e_k * dt)
        de = (e_k - e_prev) / dt if e_prev is not None else (e_k - (0.0 - xbar)) / dt
        model.subject_to(u_k == kp * e_k + ki * integral_k + kd * de)
        # implicit Euler on the nonconvex plant
        model.subject_to(x_k == x_prev + dt * (-tx * x_k * x_k + tu * u_k + td * dist))

        term = e_k * e_k * dt
        cost = term if cost is None else cost + term
        x_prev, e_prev, integral_prev = x_k, e_k, integral_k

    # the probability weight is applied by the driver, not here, so the monolith and the
    # standalone subproblems provably use the same weights.
    return cost


# ---- Family B: temporal-decomposition parameter estimation (Cao & Zavala 5.2) ----------
#
#   min_{alpha,beta,x}  sum_k int (x_k(t) - xhat_k)^2 dt
#   s.t. dx_k/dt = alpha x_k^2 + beta x_k                       (implicit Euler)
#        x_{k+1}(t_{k+1}) = x_k(t_{k+1})                        (linking)
#
# Not a stochastic program: the blocks are time partitions and the FIRST STAGE is
# (alpha, beta) plus the linking states. This is the structure that gives Cao & Zavala a
# 48-variable first stage, and it is the one that generalises to SNoGloDe's block-angular
# setting, so it is the more demanding of the two for our purposes.

_EST_STEPS_PER_BLOCK = 6


def _est_truth(t: float) -> float:
    # logistic growth: x' = a x^2 + b x with a = -0.35, b = 0.7, x(0) = 0.2
    a, b, x0 = -0.35, 0.7, 0.2
    ratio = -b / a
    return ratio / (1.0 + (ratio / x0 - 1.0) * math.exp(-b * t))


def _est_first_stage(n_blocks: int) -> list[tuple[str, float, float]]:
    spec = [("alpha", -1.0, 0.0), ("beta", 0.0, 1.5)]
    # one linking state per interior block boundary
    for k in range(1, n_blocks):
        spec.append((f"link_{k}", 0.0, 3.0))
    return spec


def _est_scenarios(n_scen: int) -> list[tuple[float, dict]]:
    rng = np.random.default_rng(20260921)
    block_len = 8.0 / n_scen
    out = []
    for k in range(n_scen):
        t0 = k * block_len
        times = [t0 + j * block_len / _EST_STEPS_PER_BLOCK for j in range(_EST_STEPS_PER_BLOCK + 1)]
        obs = [_est_truth(t) + float(rng.normal(0.0, 0.01)) for t in times]
        out.append(
            (
                1.0,
                {
                    "block": k,
                    "n_blocks": n_scen,
                    "t0": t0,
                    "dt": block_len / _EST_STEPS_PER_BLOCK,
                    "obs": obs,
                },
            )
        )
    return out


def _est_recourse(model: Model, fs: dict, data: dict, s: int):
    alpha, beta = fs["alpha"], fs["beta"]
    k, n_blocks, dt, obs = data["block"], data["n_blocks"], data["dt"], data["obs"]

    # the block's start state: block 0 starts at the known initial condition, every other
    # block starts at its incoming linking (first-stage) variable.
    x_prev = model.continuous(f"xs_{s}_0", lb=0.0, ub=3.0)
    if k == 0:
        model.subject_to(x_prev == _est_truth(0.0))
    else:
        model.subject_to(x_prev == fs[f"link_{k}"])

    cost = (x_prev - obs[0]) * (x_prev - obs[0])
    x_end = x_prev
    for j in range(1, _EST_STEPS_PER_BLOCK + 1):
        x_j = model.continuous(f"xs_{s}_{j}", lb=0.0, ub=3.0)
        model.subject_to(x_j == x_prev + dt * (alpha * x_j * x_j + beta * x_j))
        cost = cost + (x_j - obs[j]) * (x_j - obs[j])
        x_prev = x_j
        x_end = x_j

    # the block's end state must equal the outgoing linking variable
    if k + 1 < n_blocks:
        model.subject_to(x_end == fs[f"link_{k + 1}"])
    return cost


# ---- Family C: two-stage pooling / blending -------------------------------------------
#
# First stage buys two feed streams; each scenario blends them in a pool and ships to a
# quality-constrained demand. The pool-quality equality p*(y1+y2) == Q1*y1 + Q2*y2 is the
# standard pooling bilinearity -- the nonconvexity class the in-repo corpus is full of.

_POOL_Q1, _POOL_Q2 = 1.0, 3.0
_POOL_C1, _POOL_C2 = 2.2, 0.8
_POOL_PRICE = 4.0
_POOL_SHORT = 5.0


def _pool_first_stage() -> list[tuple[str, float, float]]:
    return [("q1", 0.0, 6.0), ("q2", 0.0, 6.0)]


def _pool_scenarios(n_scen: int) -> list[tuple[float, dict]]:
    rng = np.random.default_rng(20260921)
    return [
        (
            1.0 / n_scen,
            {"demand": float(rng.uniform(3.0, 9.0)), "spec": float(rng.uniform(1.6, 2.6))},
        )
        for _ in range(n_scen)
    ]


def _pool_recourse(model: Model, fs: dict, data: dict, s: int):
    q1, q2 = fs["q1"], fs["q2"]
    demand, spec = data["demand"], data["spec"]

    y1 = model.continuous(f"y1_{s}", lb=0.0, ub=6.0)
    y2 = model.continuous(f"y2_{s}", lb=0.0, ub=6.0)
    ship = model.continuous(f"ship_{s}", lb=0.0, ub=20.0)
    short = model.continuous(f"short_{s}", lb=0.0, ub=20.0)
    pq = model.continuous(f"pq_{s}", lb=_POOL_Q1, ub=_POOL_Q2)

    model.subject_to(y1 <= q1)
    model.subject_to(y2 <= q2)
    model.subject_to(ship == y1 + y2)
    model.subject_to(ship + short >= demand)
    model.subject_to(pq * ship == _POOL_Q1 * y1 + _POOL_Q2 * y2)  # pooling bilinearity
    model.subject_to(pq <= spec)

    return _POOL_SHORT * short - _POOL_PRICE * ship


FAMILIES: dict[str, Family] = {
    "pid": Family(
        "pid",
        lambda n_scen: _pid_first_stage(),
        _pid_scenarios,
        _pid_recourse,
        "PID controller tuning, Cao & Zavala 5.1; n_x = 3",
    ),
    "estimation": Family(
        "estimation",
        _est_first_stage,  # n_x = 2 + (S-1): the block count sets the first stage
        _est_scenarios,
        _est_recourse,
        "temporal-decomposition parameter estimation, Cao & Zavala 5.2; n_x = 2 + (S-1)",
    ),
    "pooling": Family(
        "pooling",
        lambda n_scen: _pool_first_stage(),
        _pool_scenarios,
        _pool_recourse,
        "two-stage pooling/blending with bilinear pool quality; n_x = 2",
        first_stage_cost=lambda fs: _POOL_C1 * fs["q1"] + _POOL_C2 * fs["q2"],
    ),
}


def first_stage_spec(family: Family, n_scen: int) -> list[tuple[str, float, float]]:
    return family.first_stage_spec(n_scen)


# --------------------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------------------


def build_monolith(family: Family, n_scen: int) -> tuple[Model, list[str]]:
    """The extensive form: one shared first stage, S recourse blocks."""
    m = Model(f"{family.name}_ef_{n_scen}")
    spec = first_stage_spec(family, n_scen)
    fs = {name: m.continuous(name, lb=lb, ub=ub) for name, lb, ub in spec}
    total = None
    for s, (prob, data) in enumerate(family.scenario_data(n_scen)):
        cost = family.add_recourse(m, fs, data, s)
        weighted = prob * cost
        total = weighted if total is None else total + weighted
    if family.first_stage_cost is not None:
        total = family.first_stage_cost(fs) + total
    m.minimize(total)
    return m, [name for name, _, _ in spec]


def build_subproblem(
    family: Family, n_scen: int, s: int, fix: dict[str, float] | None = None
) -> tuple[Model, list[str], float]:
    """Scenario ``s`` alone, with its OWN copy of the first-stage variables.

    ``fix`` pins the first stage to a candidate (the upper-bounding problem); otherwise
    the copies are free over the full box (the lower-bounding problem).
    """
    m = Model(f"{family.name}_sub_{n_scen}_{s}")
    spec = first_stage_spec(family, n_scen)
    fs = {}
    for name, lb, ub in spec:
        if fix is not None:
            v = float(min(max(fix[name], lb), ub))
            fs[name] = m.continuous(name, lb=v, ub=v)
        else:
            fs[name] = m.continuous(name, lb=lb, ub=ub)
    prob, data = family.scenario_data(n_scen)[s]
    cost = family.add_recourse(m, fs, data, s)
    if family.first_stage_cost is not None:
        # requires sum_s p_s == 1 so the weighted first-stage costs telescope back to c(x)
        assert abs(sum(p for p, _ in family.scenario_data(n_scen)) - 1.0) < 1e-9, (
            f"{family.name}: first_stage_cost needs probabilities summing to 1"
        )
        cost = family.first_stage_cost(fs) + cost
    m.minimize(prob * cost)
    return m, [name for name, _, _ in spec], prob


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


@dataclass
class CellResult:
    family: str
    scenarios: int
    n_first_stage: int
    lb_decomposed: float | None = None
    lb_decomposed_valid: bool = False
    alpha: float | None = None
    monolith_objective: float | None = None
    monolith_bound: float | None = None
    monolith_status: str | None = None
    monolith_nodes: int | None = None
    monolith_wall: float | None = None
    root_bounds: list[float] = field(default_factory=list)
    incumbent: float | None = None
    decomposed_root_gap: float | None = None
    discopt_root_gap: float | None = None
    decomposition_wins: bool | None = None
    sub_wall: float | None = None
    sub_certified: int = 0
    sub_statuses: list[str] = field(default_factory=list)
    alpha_statuses: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def _rel_gap(ub: float, lb: float) -> float:
    denom = max(abs(ub), 1e-6)
    return (ub - lb) / denom


def _solve(model: Model, time_limit: float):
    return model.solve(time_limit=time_limit, deterministic=True)


def run_cell(
    family: Family,
    n_scen: int,
    time_limit: float,
    sub_time_limit: float,
    replicates: int,
) -> CellResult:
    spec = first_stage_spec(family, n_scen)
    cell = CellResult(family=family.name, scenarios=n_scen, n_first_stage=len(spec))

    # --- arm 1: discopt on the monolithic extensive form -------------------------------
    print(f"  [{family.name} S={n_scen}] monolith ({len(spec)} first-stage vars) ...", flush=True)
    for rep in range(replicates):
        m, _ = build_monolith(family, n_scen)
        t0 = time.time()
        try:
            res = _solve(m, time_limit)
        except Exception:
            cell.failures.append(f"monolith rep{rep}: {traceback.format_exc()}")
            print(f"    monolith rep{rep} FAILED\n{traceback.format_exc()}", flush=True)
            continue
        if rep == 0:
            cell.monolith_objective = res.objective
            cell.monolith_bound = res.bound
            cell.monolith_status = str(res.status)
            cell.monolith_nodes = res.node_count
            cell.monolith_wall = time.time() - t0
        if res.root_bound is not None and math.isfinite(float(res.root_bound)):
            cell.root_bounds.append(float(res.root_bound))
        print(
            f"    rep{rep}: status={res.status} obj={res.objective} "
            f"root_bound={res.root_bound} bound={res.bound} nodes={res.node_count} "
            f"wall={time.time() - t0:.1f}s",
            flush=True,
        )

    # --- arm 2: the decomposed root bound ----------------------------------------------
    print(f"  [{family.name} S={n_scen}] {n_scen} lower-bounding subproblems ...", flush=True)
    lb_total = 0.0
    lb_valid = True
    candidates: list[dict[str, float]] = []
    weights: list[float] = []
    t_sub = time.time()
    for s in range(n_scen):
        sub, names, prob = build_subproblem(family, n_scen, s)
        try:
            res = _solve(sub, sub_time_limit)
        except Exception:
            cell.failures.append(f"sub {s}: {traceback.format_exc()}")
            print(f"    sub {s} FAILED\n{traceback.format_exc()}", flush=True)
            lb_valid = False
            break
        # CORRECTNESS: the node lower bound sums the subproblems' *dual bounds*, never
        # their incumbents. A local/incumbent value here would make LB_dec exceed the
        # true optimum -- a false bound (CLAUDE.md, correctness first).
        if res.bound is None or not res.bound_valid or not math.isfinite(float(res.bound)):
            lb_valid = False
            cell.failures.append(f"sub {s}: no valid bound (status={res.status})")
            print(f"    sub {s}: NO VALID BOUND (status={res.status})", flush=True)
            break
        lb_total += float(res.bound)
        cell.sub_statuses.append(str(res.status))
        if res.gap_certified:
            cell.sub_certified += 1
        print(
            f"    sub {s}: status={res.status} bound={res.bound:.6g} "
            f"obj={res.objective} certified={res.gap_certified}",
            flush=True,
        )
        if res.x:
            candidates.append({n: float(np.asarray(res.x[n])) for n in names if n in res.x})
            weights.append(prob)
    cell.sub_wall = time.time() - t_sub
    if lb_valid:
        cell.lb_decomposed = lb_total
        cell.lb_decomposed_valid = True
        print(
            f"    LB_dec = {lb_total:.6g}  ({cell.sub_wall:.1f}s for {n_scen} subproblems)",
            flush=True,
        )

    # --- arm 2b: the method's own upper bound (mean candidate, Cao & Zavala 2.2) --------
    if candidates and len(candidates) == n_scen:
        wsum = sum(weights) or 1.0
        xhat = {
            name: sum(w * c[name] for w, c in zip(weights, candidates, strict=True)) / wsum
            for name in candidates[0]
        }
        alpha = 0.0
        ok = True
        for s in range(n_scen):
            sub, _, _ = build_subproblem(family, n_scen, s, fix=xhat)
            try:
                res = _solve(sub, sub_time_limit)
            except Exception:
                cell.failures.append(f"ub sub {s}: {traceback.format_exc()}")
                ok = False
                break
            cell.alpha_statuses.append(str(res.status))
            if res.objective is None or not math.isfinite(float(res.objective)):
                ok = False
                cell.failures.append(f"ub sub {s}: no incumbent at xhat (status={res.status})")
                print(f"    ub sub {s}: NO INCUMBENT (status={res.status})", flush=True)
                break
            alpha += float(res.objective)
        if ok:
            cell.alpha = alpha
            print(f"    alpha (fixed-xhat UB) = {alpha:.6g}", flush=True)

    # --- compare ------------------------------------------------------------------------
    incumbents = [
        v
        for v in (cell.monolith_objective, cell.alpha)
        if v is not None and math.isfinite(float(v))
    ]
    if incumbents:
        cell.incumbent = min(float(v) for v in incumbents)
    if cell.incumbent is not None and cell.lb_decomposed is not None:
        cell.decomposed_root_gap = _rel_gap(cell.incumbent, cell.lb_decomposed)
    if cell.incumbent is not None and cell.root_bounds:
        # be generous to the baseline: use its BEST (largest) root bound across replicates
        cell.discopt_root_gap = _rel_gap(cell.incumbent, max(cell.root_bounds))
    if cell.decomposed_root_gap is not None and cell.discopt_root_gap is not None:
        cell.decomposition_wins = cell.decomposed_root_gap < cell.discopt_root_gap
    return cell


def _write_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--families", default="pid,estimation,pooling")
    ap.add_argument("--scenarios", default="4,8,16")
    ap.add_argument("--time-limit", type=float, default=60.0, help="monolith budget (s)")
    ap.add_argument("--sub-time-limit", type=float, default=30.0, help="per-subproblem budget (s)")
    ap.add_argument(
        "--replicates", type=int, default=3, help="monolith repeats, for the root-bound spread"
    )
    ap.add_argument("--out", default="reports/stochastic_evpi_entry.json")
    args = ap.parse_args()

    # rule 8: prove which code is loaded
    import discopt

    print(f"discopt.__file__ = {discopt.__file__}", flush=True)
    assert "/home/user/discopt/" in discopt.__file__, f"unexpected discopt: {discopt.__file__}"

    # rule 9: load gate
    load = os.getloadavg()
    print(f"loadavg before: {load}", flush=True)

    families = [FAMILIES[f] for f in args.families.split(",") if f]
    scen_counts = [int(s) for s in args.scenarios.split(",") if s]

    cells: list[CellResult] = []
    comparisons = 0
    t_start = time.time()
    for fam in families:
        print(f"\n=== family {fam.name}: {fam.note} ===", flush=True)
        for n_scen in scen_counts:
            cell = run_cell(fam, n_scen, args.time_limit, args.sub_time_limit, args.replicates)
            cells.append(cell)
            _write_report(
                args.out,
                {
                    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "discopt_file": discopt.__file__,
                    "args": vars(args),
                    "loadavg_before": load,
                    "partial": True,
                    "cells": [asdict(c) for c in cells],
                },
            )
            if cell.decomposition_wins is not None:
                comparisons += 1
                verdict = "DECOMP WINS" if cell.decomposition_wins else "monolith root better"
                print(
                    f"  => S={n_scen}: decomposed gap {cell.decomposed_root_gap:.4%} vs "
                    f"discopt root gap {cell.discopt_root_gap:.4%}  [{verdict}]",
                    flush=True,
                )
            else:
                print(f"  => S={n_scen}: NO COMPARISON (see failures)", flush=True)

    print(f"\nloadavg after: {os.getloadavg()}", flush=True)
    print(f"total wall: {time.time() - t_start:.1f}s", flush=True)

    # summary
    wins = sum(1 for c in cells if c.decomposition_wins)
    print("\n================ SUMMARY ================", flush=True)
    print(f"executed comparisons: {comparisons}", flush=True)
    print(f"decomposition wins:   {wins}/{comparisons}", flush=True)
    for c in cells:
        spread = f"{statistics.pstdev(c.root_bounds):.3g}" if len(c.root_bounds) > 1 else "n/a"
        print(
            f"  {c.family:<11} S={c.scenarios:<4} n_x={c.n_first_stage:<3} "
            f"dec_gap={_fmt(c.decomposed_root_gap)} root_gap={_fmt(c.discopt_root_gap)} "
            f"subs_certified={c.sub_certified}/{c.scenarios} "
            f"mono_status={c.monolith_status} root_bound_sd={spread} "
            f"sub_wall={c.sub_wall or float('nan'):.1f}s "
            f"mono_wall={(c.monolith_wall or float('nan')):.1f}s",
            flush=True,
        )

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "discopt_file": discopt.__file__,
        "args": vars(args),
        "loadavg_before": load,
        "loadavg_after": os.getloadavg(),
        "comparisons": comparisons,
        "wins": wins,
        "cells": [asdict(c) for c in cells],
    }
    _write_report(args.out, out)
    print(f"\nwrote {args.out}", flush=True)

    # rule 6: a probe that compared nothing is a failure, not a pass.
    if comparisons == 0:
        print("FAIL: zero comparisons executed -- the probe measured nothing.", flush=True)
        return 1
    return 0


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4%}"


if __name__ == "__main__":
    sys.exit(main())
