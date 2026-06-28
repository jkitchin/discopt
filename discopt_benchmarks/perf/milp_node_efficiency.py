"""Multidimensional-knapsack node-efficiency micro-bench (issue #331, Step 1).

The pure-MILP engine (``nlp_solver="simplex"`` → Rust
``crates/discopt-core/src/bnb/milp_driver.rs``, exposed as
``discopt._rust.solve_milp_py``) is *wall-clock* competitive with SCIP on
small/medium dense MILPs but explores far more **nodes**. Issue #331 Step 1 asks
us to *reproduce and attribute* that node gap before changing any solver code:

* **Reproduce** — rebuild the throwaway multidim-knapsack generator as a
  committed, deterministic bench (the ``mdk{n}x{m}`` family) and solve every
  instance both ways (discopt simplex engine + SCIP), confirming objectives
  match.
* **Attribute** — for each instance record the root LP bound, the root bound
  *after* presolve+cuts, the integrality gap closed at the root, and the node
  count; then **ablate** the ``MilpOptions`` levers (``presolve``,
  ``root_cuts``, ``cut_rounds``, ``node_cuts``, ``strong_branch``, and
  ``heuristics`` — added because it turns out to dominate node count)
  independently and record the node reduction attributable to each. This says
  whether the extra nodes come from weak *bounds* (presolve/cuts), weak
  *branching*, or the *primal heuristic* — they point to different fixes.

This module is the engine, not the solver path: it calls ``solve_milp_py``
directly so it measures the B&B driver, not the ``_SIMPLEX_MILP_BUDGET_CAP_S``
cap or the Python orchestration in ``solver.py``.

Run it::

    python -m discopt_benchmarks.perf.milp_node_efficiency               # full table
    python -m discopt_benchmarks.perf.milp_node_efficiency --quick       # small sizes only
    python -m discopt_benchmarks.perf.milp_node_efficiency --out DIR     # write md+json

SCIP (``pyscipopt``) is optional; without it the discopt-side ablation still
runs and the SCIP columns are reported as unavailable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

# ── Instance family ──────────────────────────────────────────────────────────
# (n_items, n_knapsack_constraints) — the sizes from issue #331's measured table.
MDK_SIZES: list[tuple[int, int]] = [
    (30, 5),
    (40, 5),
    (50, 8),
    (60, 8),
    (70, 10),
    (90, 12),
    (120, 15),
    (150, 20),
    (200, 25),
]

INF = 1e20


def mdk_name(n: int, m: int) -> str:
    """Canonical instance name, e.g. ``mdk30x5``."""
    return f"mdk{n}x{m}"


@dataclass
class MdkInstance:
    """A 0/1 multidimensional knapsack: ``max valueᵀx s.t. W x ≤ cap, x ∈ {0,1}``."""

    name: str
    n: int
    m: int
    weights: np.ndarray  # (m, n) integer weights per knapsack dimension
    values: np.ndarray  # (n,) item values
    cap: np.ndarray  # (m,) per-dimension capacities


def gen_mdk(n: int, m: int) -> MdkInstance:
    """Deterministically generate a 0/1 multidimensional knapsack.

    Weights are integers in ``[1, 100]``; capacities are ``floor(0.5·Σ weights)``
    (the standard half-sum ratio); values are **uncorrelated** integers in
    ``[1, 100]``, independent of the weights. This is deliberately the *easy*
    knapsack regime that issue #331 measured (it solves in hundreds–tens-of-
    thousands of nodes, sub-second) — the point of the bench is the discopt-vs-
    SCIP **node ratio** in the regime where discopt is already wall-clock
    competitive, not to build adversarially hard instances. (Strongly
    value↔weight-correlated knapsacks are the pathological B&B case and would
    blow past any sane time budget, masking the node-ratio signal.) The RNG seed
    is a pure function of ``(n, m)`` so the instance is reproducible and needs no
    committed data file.
    """
    rng = np.random.default_rng(seed=1_000 * n + m)
    weights = rng.integers(1, 101, size=(m, n)).astype(np.float64)
    cap = np.floor(0.5 * weights.sum(axis=1)).astype(np.float64)
    values = rng.integers(1, 101, size=n).astype(np.float64)
    return MdkInstance(mdk_name(n, m), n, m, weights, values, cap)


# ── Standard-form marshalling for solve_milp_py ──────────────────────────────
def _std_form(inst: MdkInstance) -> dict:
    """Build the engine's standard form ``A z = b, l ≤ z ≤ u`` (one slack/row).

    The engine *minimizes*, so we minimize ``-valueᵀx``; structural columns are
    ``[0, n)`` (the binaries) and slacks follow. Knapsack rows ``W x ≤ cap``
    become ``W x + s = cap`` with ``s ≥ 0``.
    """
    n, m = inst.n, inst.m
    a_std = np.zeros((m, n + m), dtype=np.float64)
    a_std[:, :n] = inst.weights
    a_std[:, n:] = np.eye(m)
    c_std = np.concatenate([-inst.values, np.zeros(m)])
    lb = np.concatenate([np.zeros(n), np.zeros(m)])
    ub = np.concatenate([np.ones(n), np.full(m, INF)])
    return {
        "c": np.ascontiguousarray(c_std),
        "a": np.ascontiguousarray(a_std),
        "b": np.ascontiguousarray(inst.cap),
        "lb": np.ascontiguousarray(lb),
        "ub": np.ascontiguousarray(ub),
        "int_cols": np.ascontiguousarray(np.arange(n, dtype=np.int64)),
        "n_struct": n,
    }


# ── discopt engine calls ─────────────────────────────────────────────────────
# All-off baseline: no presolve, no cuts, no strong branching, no heuristics.
# Each ablation config flips exactly one lever on top of this baseline; "full"
# turns everything on. Node reduction attributable to a lever = baseline_nodes −
# nodes(baseline + that lever).
_OFF = {
    "root_cuts": 0,
    "cut_rounds": 0,
    "node_cuts": False,
    "max_pool_cuts": 0,
    "heuristics": False,
    "presolve": False,
    "strong_branch": False,
    # New levers are set explicitly so the ablation baseline is truly all-off
    # regardless of the binding's defaults (reduced_cost_fixing defaults true).
    "reduced_cost_fixing": False,
    "sb_max_cands": 8,
    "sb_node_budget": 1024,
}


def _cfg(**overrides) -> dict:
    cfg = dict(_OFF)
    cfg.update(overrides)
    return cfg


# Ablation portfolio. Each entry layers ONE lever onto the all-off baseline so
# the marginal node reduction is cleanly attributable; `full` is the everything-on
# config (and `prod` mirrors the PyO3 production defaults).
ABLATION: dict[str, dict] = {
    "baseline": _cfg(),
    "presolve": _cfg(presolve=True),
    "root_cuts": _cfg(root_cuts=16, cut_rounds=1, max_pool_cuts=128),
    "cut_rounds": _cfg(root_cuts=64, cut_rounds=10, max_pool_cuts=256),
    "node_cuts": _cfg(node_cuts=True, max_pool_cuts=256),
    "strong_branch": _cfg(strong_branch=True),
    "heuristics": _cfg(heuristics=True),
    "reduced_cost_fixing": _cfg(heuristics=True, reduced_cost_fixing=True),
    "prod": _cfg(
        root_cuts=16,
        cut_rounds=1,
        node_cuts=False,
        max_pool_cuts=128,
        heuristics=True,
        presolve=True,
        strong_branch=True,
        reduced_cost_fixing=True,
    ),
    "full": _cfg(
        root_cuts=64,
        cut_rounds=10,
        node_cuts=True,
        max_pool_cuts=256,
        heuristics=True,
        presolve=True,
        strong_branch=True,
        reduced_cost_fixing=True,
    ),
}


@dataclass
class DiscoptRun:
    status: str
    obj: float | None
    bound: float | None
    nodes: int
    lp_iters: int
    wall_s: float


def solve_discopt(
    std: dict,
    cfg: dict,
    *,
    max_nodes: int = 2_000_000,
    gap_tol: float = 1e-4,
    time_limit_s: float = 0.0,
) -> DiscoptRun:
    """Solve via the Rust simplex B&B engine with the given lever config.

    ``obj`` is the engine's incumbent (a value-maximizing objective is reported
    here in the engine's *minimize -value* sign, so the achieved knapsack value
    is ``-obj``).
    """
    from discopt._rust import solve_milp_py

    t0 = time.perf_counter()
    status, x, obj, bound, nodes, lp_iters = solve_milp_py(
        std["c"],
        std["a"],
        std["b"],
        std["lb"],
        std["ub"],
        std["int_cols"],
        std["n_struct"],
        0.0,
        int(max_nodes),
        float(gap_tol),
        time_limit_s=float(time_limit_s),
        **cfg,
    )
    wall = time.perf_counter() - t0
    return DiscoptRun(
        status=status,
        obj=float(obj) if np.isfinite(obj) else None,
        bound=float(bound) if np.isfinite(bound) else None,
        nodes=int(nodes),
        lp_iters=int(lp_iters),
        wall_s=wall,
    )


def lp_relaxation_bound(std: dict) -> float:
    """Pure root LP relaxation objective (no integrality, no cuts, no presolve).

    Computed by calling the same engine with an empty integer set, so the LP
    bound is exactly the one the B&B root sees before any branching.
    """
    from discopt._rust import solve_milp_py

    empty = np.zeros(0, dtype=np.int64)
    # No integer columns ⇒ the root LP is the optimum and the search terminates
    # immediately; a generous node cap just avoids a spurious node-limit exit.
    _, _, obj, _, _, _ = solve_milp_py(
        std["c"],
        std["a"],
        std["b"],
        std["lb"],
        std["ub"],
        empty,
        std["n_struct"],
        0.0,
        1_000,
        1e-9,
        **_cfg(),
    )
    return float(obj)


def root_bound_after(std: dict, cfg: dict) -> float:
    """Root dual bound after presolve + root cuts, before deeper branching.

    The engine only records the global dual bound once the root node has been
    fully processed (presolve, root cut rounds, root LP re-solve) and the first
    branch created — ``max_nodes=2`` is the smallest budget that reaches that
    point, so the returned bound is the strengthened root bound and not the
    sentinel ``-inf`` that ``max_nodes=1`` leaves it at.
    """
    run = solve_discopt(std, cfg, max_nodes=2, gap_tol=1e-9)
    return run.bound if run.bound is not None else float("nan")


# ── SCIP reference ───────────────────────────────────────────────────────────
def scip_available() -> bool:
    try:
        import pyscipopt  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class ScipRun:
    nodes: int
    obj: float | None
    root_dualbound: float | None
    wall_s: float
    version: str
    status: str = "?"


def _build_scip(inst: MdkInstance):
    """Construct the SCIP model for `inst` (max valueᵀx s.t. W x ≤ cap, binary)."""
    import pyscipopt
    from pyscipopt import quicksum

    model = pyscipopt.Model(inst.name)
    model.hideOutput()
    x = [model.addVar(vtype="B", name=f"x{i}") for i in range(inst.n)]
    model.setObjective(quicksum(float(inst.values[i]) * x[i] for i in range(inst.n)), "maximize")
    for j in range(inst.m):
        model.addCons(
            quicksum(float(inst.weights[j, i]) * x[i] for i in range(inst.n)) <= float(inst.cap[j])
        )
    return model


def _scip_root_bound(inst: MdkInstance) -> float | None:
    """SCIP's dual bound after processing **only the root node** (no restarts).

    Measured apples-to-apples with discopt's root bound: cap the search at one
    node and disable restarts, so the value is the bound SCIP's root presolve +
    cut rounds achieve *before branching* — not the post-restart, post-branching
    figure ``getDualboundRoot()`` reports on a full solve (which re-strengthens
    the root and inflates the apparent root gap closed).
    """
    try:
        model = _build_scip(inst)
        model.setParam("presolving/maxrestarts", 0)
        model.setParam("limits/nodes", 1)
        model.optimize()
        return float(model.getDualbound())
    except Exception:
        return None


def solve_scip(inst: MdkInstance, *, time_limit_s: float = 0.0) -> ScipRun | None:
    """Solve the identical instance with SCIP at default settings.

    Built directly from the same ``(W, values, cap)`` data the discopt path uses
    (a literal MPS round-trip would yield the same model), so node counts and
    objectives are an apples-to-apples reference. ``root_dualbound`` is SCIP's
    *true* root bound (one node, no restarts) — see :func:`_scip_root_bound`.
    """
    try:
        import pyscipopt  # noqa: F401
    except Exception:
        return None

    model = _build_scip(inst)
    if time_limit_s > 0:
        model.setParam("limits/time", float(time_limit_s))
    t0 = time.perf_counter()
    model.optimize()
    wall = time.perf_counter() - t0
    obj = float(model.getObjVal()) if model.getNSols() > 0 else None
    return ScipRun(
        nodes=int(model.getNNodes()),
        obj=obj,
        root_dualbound=_scip_root_bound(inst),
        wall_s=wall,
        version=_scip_version(),
        status=str(model.getStatus()),
    )


def _scip_version() -> str:
    try:
        import pyscipopt

        return str(pyscipopt.scip.Model().version())
    except Exception:
        return "?"


# ── Orchestration ────────────────────────────────────────────────────────────
def _gap_closed(z_lp: float, root_bound: float, z_opt: float) -> float | None:
    """Fraction of the integrality gap closed at the root: (root−LP)/(opt−LP)."""
    denom = z_opt - z_lp
    if abs(denom) < 1e-9:
        return 1.0
    if not (np.isfinite(z_lp) and np.isfinite(root_bound) and np.isfinite(z_opt)):
        return None
    return (root_bound - z_lp) / denom


@dataclass
class InstanceResult:
    name: str
    n: int
    m: int
    # objectives (knapsack value, i.e. maximize sense)
    opt_value: float | None = None
    scip_value: float | None = None
    objectives_match: bool | None = None
    # discopt minimize-sense bounds for gap analysis
    z_lp: float | None = None
    z_opt: float | None = None
    discopt_root_bound: float | None = None
    discopt_root_gap_closed: float | None = None
    scip_root_gap_closed: float | None = None
    # node + wall comparison at production config
    discopt_status: str | None = None
    discopt_nodes: int | None = None
    discopt_wall_s: float | None = None
    scip_status: str | None = None
    scip_nodes: int | None = None
    scip_wall_s: float | None = None
    node_ratio: float | None = None  # only when BOTH solvers proved optimal
    # per-lever ablation: cfg name -> {nodes, wall_s, status}
    ablation: dict = field(default_factory=dict)


def run_instance(
    inst: MdkInstance,
    *,
    max_nodes: int = 2_000_000,
    time_limit_s: float = 30.0,
    do_scip: bool = True,
) -> InstanceResult:
    std = _std_form(inst)
    res = InstanceResult(name=inst.name, n=inst.n, m=inst.m)

    # Reference optimum from the production config (proven optimal expected).
    prod = solve_discopt(std, ABLATION["prod"], max_nodes=max_nodes, time_limit_s=time_limit_s)
    res.z_opt = prod.bound
    res.opt_value = -prod.obj if prod.obj is not None else None
    res.discopt_status = prod.status
    res.discopt_nodes = prod.nodes
    res.discopt_wall_s = prod.wall_s

    # Root bound decomposition (minimize sense).
    res.z_lp = lp_relaxation_bound(std)
    res.discopt_root_bound = root_bound_after(std, ABLATION["prod"])
    if res.z_opt is not None:
        res.discopt_root_gap_closed = _gap_closed(res.z_lp, res.discopt_root_bound, res.z_opt)

    # SCIP reference.
    if do_scip and scip_available():
        sc = solve_scip(inst, time_limit_s=time_limit_s)
        if sc is not None:
            res.scip_status = sc.status
            res.scip_nodes = sc.nodes
            res.scip_wall_s = sc.wall_s
            res.scip_value = sc.obj
            both_optimal = prod.status == "optimal" and sc.status == "optimal"
            # Only assert an objective match when BOTH solvers proved optimality;
            # at a shared time cap each holds a different (sound) incumbent, which
            # is not a disagreement about the optimum.
            if both_optimal and res.opt_value is not None and sc.obj is not None:
                res.objectives_match = abs(res.opt_value - sc.obj) <= 1e-6 * (1 + abs(sc.obj))
            # SCIP root dual bound is in maximize sense (value); convert to the
            # engine's minimize (-value) sense for a common gap-closed metric.
            # When SCIP closes the instance in presolve it never records a root
            # LP and returns its ±1e20 infinity sentinel — skip those (the node
            # column already shows the ≤1-node presolve solve).
            if (
                sc.root_dualbound is not None
                and abs(sc.root_dualbound) < 1e19
                and res.z_opt is not None
            ):
                scip_root_min = -sc.root_dualbound
                res.scip_root_gap_closed = _gap_closed(res.z_lp, scip_root_min, res.z_opt)
            if both_optimal and res.scip_nodes:
                res.node_ratio = res.discopt_nodes / max(1, res.scip_nodes)

    # Per-lever ablation (node reduction attributable to each lever).
    for cfg_name, cfg in ABLATION.items():
        run = solve_discopt(std, cfg, max_nodes=max_nodes, time_limit_s=time_limit_s)
        res.ablation[cfg_name] = {
            "nodes": run.nodes,
            "wall_s": run.wall_s,
            "status": run.status,
        }
    return res


def run(
    sizes: list[tuple[int, int]] | None = None,
    *,
    max_nodes: int = 2_000_000,
    time_limit_s: float = 30.0,
    do_scip: bool = True,
) -> list[InstanceResult]:
    sizes = sizes or MDK_SIZES
    out: list[InstanceResult] = []
    for n, m in sizes:
        inst = gen_mdk(n, m)
        out.append(
            run_instance(inst, max_nodes=max_nodes, time_limit_s=time_limit_s, do_scip=do_scip)
        )
    return out


# ── Reporting ────────────────────────────────────────────────────────────────
def _fmt(v, spec="{}"):
    return "—" if v is None else spec.format(v)


def to_markdown(results: list[InstanceResult]) -> str:
    lines: list[str] = []
    lines.append("# MILP node-efficiency bench (issue #331, Step 1)\n")
    lines.append(f"SCIP version: `{_scip_version()}` · discopt config: `prod` defaults\n")

    # Reproduce table. A trailing `*` on a node count flags a non-optimal exit
    # (the per-solve time/node cap); node ratio is reported only when BOTH
    # solvers proved optimality, so it is an apples-to-apples nodes-to-proof.
    lines.append("## Reproduce — discopt vs SCIP (production config)\n")
    lines.append(
        "| instance | discopt nodes | discopt wall | SCIP nodes | SCIP wall | "
        "node ratio | obj match |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        dflag = "" if r.discopt_status == "optimal" else "*"
        sflag = "" if r.scip_status == "optimal" else "*"
        dn = f"{_fmt(r.discopt_nodes)}{dflag if r.discopt_nodes is not None else ''}"
        sn = f"{_fmt(r.scip_nodes)}{sflag if r.scip_nodes is not None else ''}"
        match = "n/a" if r.objectives_match is None else _fmt(r.objectives_match)
        lines.append(
            f"| {r.name} | {dn} | {_fmt(r.discopt_wall_s, '{:.3f}s')} | "
            f"{sn} | {_fmt(r.scip_wall_s, '{:.3f}s')} | "
            f"{_fmt(r.node_ratio, '{:.1f}x')} | {match} |"
        )
    lines.append(
        "\n`*` = hit per-solve time/node cap (not proven optimal); "
        "`obj match = n/a` when a solver did not prove optimality.\n"
    )

    # Attribute — root gap closed.
    lines.append("\n## Attribute — root bound & integrality gap closed\n")
    lines.append(
        "| instance | z_LP | z_opt | discopt root bound | discopt gap closed | SCIP gap closed |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.name} | {_fmt(r.z_lp, '{:.2f}')} | {_fmt(r.z_opt, '{:.2f}')} | "
            f"{_fmt(r.discopt_root_bound, '{:.2f}')} | "
            f"{_fmt(r.discopt_root_gap_closed, '{:.1%}')} | "
            f"{_fmt(r.scip_root_gap_closed, '{:.1%}')} |"
        )

    # Ablation — nodes per lever.
    cfg_names = list(ABLATION.keys())
    lines.append("\n## Ablation — node count per lever (one lever on top of baseline)\n")
    lines.append("| instance | " + " | ".join(cfg_names) + " |")
    lines.append("|---" * (len(cfg_names) + 1) + "|")
    for r in results:
        cells = [r.name]
        for c in cfg_names:
            a = r.ablation.get(c, {})
            nodes = a.get("nodes")
            flag = "" if a.get("status") == "optimal" else "*"
            cells.append(f"{_fmt(nodes)}{flag}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append(
        "\n`*` = config hit node/time limit (not proven optimal). For capped rows "
        "the node count is *nodes explored within the budget*, not nodes-to-proof, "
        "so it conflates node-efficiency with per-node cost (stronger cuts make "
        "each node more expensive); read attribution primarily from the uncapped "
        "rows and cross-check wall time.\n"
    )

    # Ablation — node reduction vs baseline.
    lines.append("## Ablation — node reduction attributable to each lever (vs baseline)\n")
    levers = [c for c in cfg_names if c != "baseline"]
    lines.append("| instance | " + " | ".join(levers) + " |")
    lines.append("|---" * (len(levers) + 1) + "|")
    for r in results:
        base = r.ablation.get("baseline", {}).get("nodes")
        cells = [r.name]
        for c in levers:
            nodes = r.ablation.get(c, {}).get("nodes")
            if base and nodes is not None and base > 0:
                cells.append(f"{(base - nodes) / base:.0%}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_json(results: list[InstanceResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small sizes only (fast smoke)")
    ap.add_argument("--no-scip", action="store_true", help="skip SCIP reference")
    ap.add_argument("--max-nodes", type=int, default=2_000_000)
    ap.add_argument("--time-limit", type=float, default=30.0, help="per-solve wall cap (s)")
    ap.add_argument("--out", type=str, default=None, help="dir to write report.md / report.json")
    args = ap.parse_args(argv)

    sizes = MDK_SIZES[:4] if args.quick else MDK_SIZES
    results = run(
        sizes,
        max_nodes=args.max_nodes,
        time_limit_s=args.time_limit,
        do_scip=not args.no_scip,
    )
    md = to_markdown(results)
    print(md)
    if args.out:
        import os

        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "milp_node_efficiency.md"), "w") as f:
            f.write(md)
        with open(os.path.join(args.out, "milp_node_efficiency.json"), "w") as f:
            f.write(to_json(results))
        print(f"\nwrote report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
