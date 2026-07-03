"""T2.1 entry experiment: offline root-loop replay with per-stage attribution.

This is the Phase-2 *entry experiment / kill criterion* of the certification-gap
plan (`docs/dev/certification-gap-plan.md` §14, subsection "T2.1"). It does NOT
change the solver. It replays, offline and deterministically, the *cutoff-aware
root reduction loop* that Phase 2 proposes to build (T2.3), attributing the root
dual-bound movement to each candidate stage marginally, so the plan can decide
(a) which stages earn a place in the loop and (b) whether the loop's premise
survives at all.

Per-instance procedure (mirrors the spec exactly):

  1. Harvest an incumbent honestly — a short discopt solve (default 8 s) — and
     record it (or "none found").
  2. Baseline root state — today's root sequence once, with **no** incumbent
     cutoff (Rust presolve -> FBBT -> structural ``obbt_tighten_root``, matching
     solver.py:3835's cutoff-free root OBBT) — and record the baseline root
     McCormick-LP bound / relative root gap (T0.1 semantics).
  3. Replay the loop, deterministic stage order, each stage measured
     *marginally* per iteration:
        S1  presolve()   — ordered Rust passes incl. probing
        S2  fbbt_with_cutoff(incumbent)
        S3  obbt_tighten_root(incumbent_cutoff=...)   (DBBT runs inside)
        S4  envelope re-derivation (build via the relaxer at the tightened box)
            + re-separation (capture the cut pool via
            solve_at_node(..., separate=True, out_cuts=...))  -> root bound
     iterate until the bound moves < tol or the loop budget is spent.
  4. Record per stage/iteration: root gap, wall, sum of log bound-widths (box
     log-volume), #bounds tightened.
  5. Projected tree effect — apply the final tightened box as variable bounds
     and re-solve at a fixed budget (60 s); record node_count/status/objective
     vs an untightened re-solve at the same budget.
  6. Oracle-cutoff variant (diagnostic only) — repeat the loop with
     cutoff = known optimum + tol, bounding the best case reduction can deliver.

Soundness instrumentation is inline: after **every** stage, ``assert_bound_sound``
against the known-optimum oracle (bound <= oracle + tol for the internally
minimized sense). Any violation is a **P0 STOP** (§0.6): an existing reduction
pass is unsound — the script prints a prominent banner and aborts that instance
without continuing to the next stage.

House style per t11/t12: task-ID docstring; ``JAX_PLATFORMS=cpu`` +
``JAX_ENABLE_X64=1`` set BEFORE importing discopt; standalone runnable
(``python discopt_benchmarks/scripts/t21_root_loop_replay.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import math
import time
import traceback

import discopt.modeling as dm
import numpy as np
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.obbt import obbt_tighten_root
from discopt._jax.presolve_pipeline import (
    propagate_bounds_to_model,
    run_root_presolve,
)
from discopt._rust import model_to_repr

# ---------------------------------------------------------------------------
# Panel: the 20 worst global50 instances by wall ratio vs BARON
# (results/baron_vs_discopt_global50_20260618T033058.json). The first six are
# the *uncertified* substantive tail (feasible / time_limit at budget); the
# rest are certified-but-slow.
# ---------------------------------------------------------------------------
UNCERTIFIED = ["tspn05", "tanksize", "casctanks", "tls2", "st_e36", "nvs05"]
CERTIFIED_SLOW = [
    "clay0303hfsg",
    "st_e38",
    "st_test1",
    "st_testgr3",
    "st_miqp2",
    "st_miqp5",
    "m3",
    "cvxnonsep_nsig30",
    "st_miqp4",
    "st_miqp1",
    "cvxnonsep_psig40r",
    "fac2",
    "cvxnonsep_psig30",
    "flay03m",
]
PANEL = UNCERTIFIED + CERTIFIED_SLOW

# Known optima (the oracle). From minlplib.solu (=opt=/=best=) with
# docs/dev/data/cert-optima.json as cross-check. All panel instances normalize
# to MINIMIZE via the .nl reader, so these are lower-bound oracles directly.
ORACLE: dict[str, float] = {
    "tspn05": 191.2552078,  # =best= (tspn05 has only best/bestdual)
    "tanksize": 1.268643754,
    "casctanks": 9.163479388,  # =best=
    "tls2": 5.3,
    "st_e36": -246.0,
    "nvs05": 5.470934108225147,
    "clay0303hfsg": 26669.10957,
    "st_e38": 7197.727149,
    "st_test1": 0.0,
    "st_testgr3": -20.59,
    "st_miqp2": 2.0,
    "st_miqp5": -333.8888889,
    "m3": 37.8,
    "cvxnonsep_nsig30": 130.6287126,
    "st_miqp4": -4574.0,
    "st_miqp1": 281.0,
    "cvxnonsep_psig40r": 86.5451047,
    "fac2": 331837498.2,
    "cvxnonsep_psig30": 78.99885434,
    "flay03m": 48.98979486,
}

# global50 config default. discopt reports a valid dual (lower) bound in the
# internally minimized sense; the oracle is that sense too.
SND_TOL = 1e-4  # soundness slack (relative-scaled below)
LOOP_TOL = 1e-6  # loop bound-movement stop tolerance
DEFAULT_SNAPSHOT = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")


class P0StopError(Exception):
    """Raised when a stage produces an unsound bound (bound > oracle + tol)."""


# ---------------------------------------------------------------------------
# Instance resolution (mirrors runner.py:382 order, snapshot-first for panel).
# ---------------------------------------------------------------------------
def resolve_nl(name: str, snapshot: str) -> str | None:
    p = os.path.join(snapshot, f"{name}.nl")
    if os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for sub in ("python/tests/data/minlplib", "python/tests/data/minlplib_nl"):
        cand = os.path.join(root, sub, f"{name}.nl")
        if os.path.exists(cand):
            return cand
    return None


# ---------------------------------------------------------------------------
# Flat <-> block bound plumbing (matches _extract_variable_info ordering).
# ---------------------------------------------------------------------------
def flat_bounds(model) -> tuple[np.ndarray, np.ndarray]:
    lb_parts, ub_parts = [], []
    for v in model._variables:
        lb_parts.append(np.asarray(v.lb, dtype=np.float64).ravel())
        ub_parts.append(np.asarray(v.ub, dtype=np.float64).ravel())
    lb = np.concatenate(lb_parts) if lb_parts else np.zeros(0)
    ub = np.concatenate(ub_parts) if ub_parts else np.zeros(0)
    return lb, ub


def set_block_bounds(model, lb: np.ndarray, ub: np.ndarray) -> None:
    off = 0
    for v in model._variables:
        n = v.size
        sh = np.asarray(v.lb).shape
        v.lb = np.asarray(lb[off : off + n], dtype=np.float64).reshape(sh)
        v.ub = np.asarray(ub[off : off + n], dtype=np.float64).reshape(sh)
        off += n


def block_fbbt_to_flat(
    model, blk_lb: np.ndarray, blk_ub: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect per-block FBBT bounds into a flat box (only scalar blocks)."""
    tl, tu = lb.copy(), ub.copy()
    if len(blk_lb) != len(model._variables) or len(blk_ub) != len(model._variables):
        return tl, tu
    off = 0
    for bi, v in enumerate(model._variables):
        if v.size != 1:
            off += v.size
            continue
        blo, bhi = float(blk_lb[bi]), float(blk_ub[bi])
        if np.isfinite(blo):
            tl[off] = max(tl[off], blo)
        if np.isfinite(bhi):
            tu[off] = min(tu[off], bhi)
        off += 1
    return tl, tu


def log_volume(lb: np.ndarray, ub: np.ndarray) -> float:
    """Sum of log bound-widths over finite, positive-width dimensions (nats)."""
    w = ub - lb
    finite = np.isfinite(w) & (w > 1e-12)
    if not np.any(finite):
        return 0.0
    return float(np.sum(np.log(w[finite])))


def n_strictly_tightened(lb0: np.ndarray, ub0: np.ndarray, lb1: np.ndarray, ub1: np.ndarray) -> int:
    return int(np.sum(lb1 > lb0 + 1e-10) + np.sum(ub1 < ub0 - 1e-10))


# ---------------------------------------------------------------------------
# Root McCormick-LP bound at a box, capturing the separated cut pool (= S4).
# ---------------------------------------------------------------------------
def root_lp_bound(model, lb: np.ndarray, ub: np.ndarray) -> tuple[float | None, int]:
    """Cold-build the McCormick relaxation at (lb,ub), separate, return the LP
    lower bound and the number of captured cut rows. ``None`` when the model has
    no relaxable nonlinearity or the LP does not solve to optimality."""
    rel = MccormickLPRelaxer(model, build_incremental=False)
    if not rel.has_relaxable_nonlinearity:
        return None, 0
    cuts: list = []
    res = rel.solve_at_node(lb, ub, separate=True, out_cuts=cuts)
    n_cut_rows = 0
    for entry in cuts:
        try:
            a_rows, _b = entry
            n_cut_rows += int(np.asarray(a_rows).shape[0])
        except Exception:
            n_cut_rows += 1
    if res.status != "optimal" or res.lower_bound is None:
        return None, n_cut_rows
    return float(res.lower_bound), n_cut_rows


def rel_gap(bound: float | None, incumbent: float | None) -> float | None:
    """Relative root gap = |incumbent - bound| / max(1, |incumbent|)."""
    if bound is None or incumbent is None:
        return None
    return abs(incumbent - bound) / max(1.0, abs(incumbent))


def assert_sound(name: str, stage: str, bound: float | None, oracle: float) -> None:
    """P0 STOP if the (minimization) dual bound exceeds the oracle by > tol.

    Tolerance is relative-scaled to the oracle magnitude (same spirit as
    utils/soundness.assert_bound_sound, but usable on a single scalar bound).
    """
    if bound is None:
        return
    tol = SND_TOL * max(1.0, abs(oracle))
    if bound > oracle + tol:
        raise P0StopError(
            f"{name} / {stage}: dual bound {bound:.10g} EXCEEDS known optimum "
            f"{oracle:.10g} by {bound - oracle:.3g} (> tol {tol:.3g}) — "
            f"an existing reduction pass is UNSOUND (false certificate)."
        )


# ---------------------------------------------------------------------------
# The four stages, each applied to a (lb,ub) box; return the new box.
# ---------------------------------------------------------------------------
def stage_s1_presolve(model, lb, ub) -> tuple[np.ndarray, np.ndarray]:
    set_block_bounds(model, lb, ub)
    mr = model_to_repr(model, getattr(model, "_builder", None))
    mr2, _stats = run_root_presolve(mr)
    propagate_bounds_to_model(model, mr2)  # writes tightened bounds back into blocks
    return flat_bounds(model)


def stage_s2_fbbt_cutoff(model, lb, ub, incumbent) -> tuple[np.ndarray, np.ndarray]:
    set_block_bounds(model, lb, ub)
    mr = model_to_repr(model, getattr(model, "_builder", None))
    flb, fub = mr.fbbt_with_cutoff(max_iter=10, tol=1e-8, incumbent_bound=float(incumbent))
    return block_fbbt_to_flat(model, np.asarray(flb), np.asarray(fub), lb, ub)


def stage_s3_obbt(model, lb, ub, incumbent, budget_s) -> tuple[np.ndarray, np.ndarray, int]:
    res = obbt_tighten_root(
        model,
        lb,
        ub,
        rounds=3,
        deadline=time.perf_counter() + budget_s,
        incumbent_cutoff=None if incumbent is None else float(incumbent),
    )
    if res.n_tightened <= 0:
        return lb, ub, 0
    nlb = np.maximum(lb, res.lb)
    nub = np.minimum(ub, res.ub)
    return nlb, nub, int(res.n_tightened)


# ---------------------------------------------------------------------------
# Per-instance replay.
# ---------------------------------------------------------------------------
def replay_instance(
    name: str,
    path: str,
    *,
    harvest_s: float,
    loop_budget_s: float,
    obbt_stage_s: float,
    resolve_s: float,
    max_iters: int,
    oracle_variant: bool,
) -> dict:
    result: dict = {"name": name, "status": "ok"}
    oracle = ORACLE.get(name)
    if oracle is None:
        result["status"] = "no_oracle"
        return result

    # ---- 1. Harvest incumbent honestly ------------------------------------
    model = dm.from_nl(path)
    lb0, ub0 = flat_bounds(model)
    t = time.perf_counter()
    solve = model.solve(time_limit=harvest_s)
    result["harvest_wall"] = time.perf_counter() - t
    result["harvest_status"] = str(solve.status)
    incumbent = None
    if solve.objective is not None and math.isfinite(solve.objective):
        incumbent = float(solve.objective)
    result["incumbent"] = incumbent
    result["solve_bound"] = solve.bound
    result["solve_nodes"] = solve.node_count
    result["certified"] = str(solve.status) == "optimal"

    if incumbent is None:
        # No incumbent -> the cutoff stages (S2/S3 with cutoff) have no cutoff;
        # fall back to the oracle cutoff for the loop so the experiment still
        # produces attribution, but flag it clearly (spec risk 3).
        result["incumbent_source"] = "none_found->oracle"
        incumbent = oracle
    else:
        result["incumbent_source"] = "harvested"

    # ---- 2. Baseline root state (today's cutoff-FREE root sequence) --------
    # Fresh model so the harvest solve cannot have mutated blocks.
    model = dm.from_nl(path)
    lb, ub = flat_bounds(model)
    # S1 presolve (no cutoff)
    lb, ub = stage_s1_presolve(model, lb, ub)
    # S2-analogue at baseline is FBBT *without* a cutoff -> plain root FBBT via
    # presolve already ran fbbt; the structural root OBBT below is cutoff-free.
    # S3 structural OBBT, NO cutoff (matches solver.py:3891).
    lb, ub, _ = stage_s3_obbt(model, lb, ub, None, obbt_stage_s)
    base_bound, base_cuts = root_lp_bound(model, lb, ub)
    assert_sound(name, "baseline", base_bound, oracle)
    base_gap = rel_gap(base_bound, incumbent)
    result["baseline_bound"] = base_bound
    result["baseline_gap"] = base_gap
    result["baseline_logvol"] = log_volume(lb, ub)
    result["baseline_cut_rows"] = base_cuts
    base_lb, base_ub = lb.copy(), ub.copy()

    # ---- 3. Replay the cutoff-aware loop, marginal per-stage attribution ---
    def run_loop(cutoff: float, tag: str) -> dict:
        m = dm.from_nl(path)
        clb, cub = base_lb.copy(), base_ub.copy()
        cur_bound = base_bound
        stage_rows: list[dict] = []
        deadline = time.perf_counter() + loop_budget_s
        it = 0
        while it < max_iters and time.perf_counter() < deadline:
            it += 1
            prev_iter_bound = cur_bound
            for sid, fn in (
                ("S1_presolve", "s1"),
                ("S2_fbbt_cutoff", "s2"),
                ("S3_obbt", "s3"),
                ("S4_reseparate", "s4"),
            ):
                pre_bound = cur_bound
                pre_gap = rel_gap(pre_bound, cutoff)
                pre_lb, pre_ub = clb.copy(), cub.copy()
                t0 = time.perf_counter()
                n_tight = 0
                if fn == "s1":
                    clb, cub = stage_s1_presolve(m, clb, cub)
                elif fn == "s2":
                    clb, cub = stage_s2_fbbt_cutoff(m, clb, cub, cutoff)
                elif fn == "s3":
                    clb, cub, n_tight = stage_s3_obbt(m, clb, cub, cutoff, obbt_stage_s)
                # S4: re-derive envelope at the (possibly tightened) box and
                # re-separate; this is where the marginal bound is re-measured.
                new_bound, n_cuts = root_lp_bound(m, clb, cub)
                wall = time.perf_counter() - t0
                # A stage's own marginal bound move is measured by the root LP
                # bound recomputed after it (S1-S3), and for S4 the re-separation
                # itself. Use the post-stage bound as cur_bound.
                if new_bound is not None:
                    cur_bound = max(pre_bound, new_bound) if pre_bound is not None else new_bound
                assert_sound(name, f"{tag}/it{it}/{sid}", cur_bound, oracle)
                post_gap = rel_gap(cur_bound, cutoff)
                n_tight_box = n_strictly_tightened(pre_lb, pre_ub, clb, cub)
                # marginal relative root-gap movement of THIS stage
                marg = None
                if pre_gap is not None and post_gap is not None and pre_gap > 1e-12:
                    marg = (pre_gap - post_gap) / pre_gap
                stage_rows.append(
                    {
                        "iter": it,
                        "stage": sid,
                        "pre_gap": pre_gap,
                        "post_gap": post_gap,
                        "marginal_relgap_move": marg,
                        "bound": cur_bound,
                        "wall": wall,
                        "n_tightened": max(n_tight, n_tight_box),
                        "cut_rows": n_cuts,
                        "logvol": log_volume(clb, cub),
                    }
                )
            # loop convergence: bound moved < tol this whole iteration
            if (
                cur_bound is not None
                and prev_iter_bound is not None
                and abs(cur_bound - prev_iter_bound) < LOOP_TOL
            ):
                break
        final_gap = rel_gap(cur_bound, cutoff)
        gap_reduction = None
        if base_gap is not None and base_gap > 1e-12 and final_gap is not None:
            gap_reduction = (base_gap - final_gap) / base_gap
        return {
            "cutoff": cutoff,
            "iters": it,
            "final_bound": cur_bound,
            "final_gap": final_gap,
            "gap_reduction": gap_reduction,
            "final_lb": clb,
            "final_ub": cub,
            "stage_rows": stage_rows,
            "final_logvol": log_volume(clb, cub),
        }

    loop = run_loop(incumbent, "loop")
    result["loop"] = {k: v for k, v in loop.items() if k not in ("final_lb", "final_ub")}

    # ---- 5. Projected tree effect: tightened re-solve vs untightened -------
    tree_budget = 60.0
    # untightened baseline re-solve
    m_un = dm.from_nl(path)
    t = time.perf_counter()
    r_un = m_un.solve(time_limit=tree_budget)
    result["tree_untightened"] = {
        "status": str(r_un.status),
        "nodes": r_un.node_count,
        "objective": r_un.objective,
        "wall": time.perf_counter() - t,
    }
    # tightened re-solve
    m_ti = dm.from_nl(path)
    set_block_bounds(m_ti, loop["final_lb"], loop["final_ub"])
    t = time.perf_counter()
    r_ti = m_ti.solve(time_limit=tree_budget)
    result["tree_tightened"] = {
        "status": str(r_ti.status),
        "nodes": r_ti.node_count,
        "objective": r_ti.objective,
        "wall": time.perf_counter() - t,
    }
    # tightened re-solve must reproduce the reference objective to tol
    if r_ti.objective is not None and math.isfinite(r_ti.objective):
        tol = SND_TOL * max(1.0, abs(oracle))
        if r_ti.objective < oracle - tol:
            raise P0StopError(
                f"{name}: tightened-box re-solve objective {r_ti.objective:.10g} "
                f"below oracle {oracle:.10g} (> tol) — reduction cut the optimum."
            )
    closed_within_10 = (
        str(r_ti.status) == "optimal"
        and (r_ti.node_count or 0) <= 10
        and not (str(r_un.status) == "optimal" and (r_un.node_count or 0) <= 10)
    )
    result["closed_within_10_by_reduction"] = bool(closed_within_10)

    # ---- 6. Oracle-cutoff diagnostic variant ------------------------------
    if oracle_variant:
        oc = run_loop(oracle + SND_TOL * max(1.0, abs(oracle)), "oracle")
        result["oracle_loop"] = {
            k: v for k, v in oc.items() if k not in ("final_lb", "final_ub", "stage_rows")
        }
    return result


# ---------------------------------------------------------------------------
# Kill-criterion evaluation over the panel results.
# ---------------------------------------------------------------------------
def evaluate_kill(results: list[dict]) -> dict:
    stages = ["S1_presolve", "S2_fbbt_cutoff", "S3_obbt", "S4_reseparate"]
    # per-stage: max marginal relgap move over all (instance, iteration) rows.
    per_stage_max: dict[str, float] = dict.fromkeys(stages, 0.0)
    per_stage_any_ge5: dict[str, bool] = dict.fromkeys(stages, False)
    for r in results:
        loop = r.get("loop")
        if not loop:
            continue
        for row in loop["stage_rows"]:
            m = row["marginal_relgap_move"]
            if m is None:
                continue
            s = row["stage"]
            per_stage_max[s] = max(per_stage_max[s], m)
            if m >= 0.05:
                per_stage_any_ge5[s] = True
    include = [s for s in stages if per_stage_any_ge5[s]]
    exclude = [s for s in stages if not per_stage_any_ge5[s]]

    # loop-level (a): median relative root-gap reduction over the six uncertified.
    unc_reductions = []
    for r in results:
        if r["name"] in UNCERTIFIED and r.get("loop"):
            gr = r["loop"].get("gap_reduction")
            if gr is not None:
                unc_reductions.append(gr)
    median_unc = float(np.median(unc_reductions)) if unc_reductions else None

    # loop-level (b): fraction of tree-opening certified instances closing <=10 nodes.
    cert_opening = [r for r in results if r["name"] in CERTIFIED_SLOW and r.get("tree_untightened")]
    n_closed = sum(1 for r in cert_opening if r.get("closed_within_10_by_reduction"))
    frac_closed = (n_closed / len(cert_opening)) if cert_opening else None

    crit_a = median_unc is not None and median_unc >= 0.25
    crit_b = frac_closed is not None and frac_closed >= 0.30
    survives = crit_a or crit_b
    return {
        "per_stage_max": per_stage_max,
        "per_stage_any_ge5pct": per_stage_any_ge5,
        "include": include,
        "exclude": exclude,
        "median_unc_gap_reduction": median_unc,
        "n_unc_measured": len(unc_reductions),
        "frac_certified_closed_10": frac_closed,
        "n_certified_opening": len(cert_opening),
        "crit_a_median_ge25pct": crit_a,
        "crit_b_frac_ge30pct": crit_b,
        "premise_survives": survives,
    }


def fmt_pct(x) -> str:
    return "   n/a" if x is None else f"{100 * x:6.1f}%"


def print_summary(results: list[dict], kill: dict, ran: list[str], skipped: list[str]) -> None:
    print("\n" + "=" * 78)
    print("T2.1 ROOT-LOOP REPLAY — SUMMARY")
    print("=" * 78)
    print(f"\nRan {len(ran)}/{len(PANEL)} panel instances: {', '.join(ran)}")
    if skipped:
        print(
            f"SKIPPED {len(skipped)} (clearly labeled, NOT silently dropped): {', '.join(skipped)}"
        )

    print("\n--- Per-instance loop outcome ---")
    hdr = (
        f"{'instance':18s} {'cert?':5s} {'base_gap':>10s} {'final_gap':>10s} "
        f"{'gap_red':>8s} {'nodes_un':>9s} {'nodes_ti':>9s}"
    )
    print(hdr)
    for r in results:
        loop = r.get("loop", {})
        tu = r.get("tree_untightened", {})
        ti = r.get("tree_tightened", {})
        bg = r.get("baseline_gap")
        fg = loop.get("final_gap")
        bg = float("nan") if bg is None else bg
        fg = float("nan") if fg is None else fg
        print(
            f"{r['name']:18s} "
            f"{('Y' if r.get('certified') else 'n'):5s} "
            f"{bg:10.4f} {fg:10.4f} "
            f"{fmt_pct(loop.get('gap_reduction')):>8s} "
            f"{str(tu.get('nodes')):>9s} {str(ti.get('nodes')):>9s}"
        )

    print("\n--- Per-stage marginal attribution (max marginal relgap move over panel) ---")
    for s, v in kill["per_stage_max"].items():
        flag = "INCLUDE" if kill["per_stage_any_ge5pct"][s] else "DROP (<5% everywhere)"
        print(f"  {s:16s}  max_marginal={fmt_pct(v)}   -> {flag}")

    print("\n--- KILL CRITERION ---")
    print(f"  include stages: {kill['include']}")
    print(f"  exclude stages: {kill['exclude']}")
    print(
        f"  (a) median relgap reduction on {kill['n_unc_measured']} uncertified = "
        f"{fmt_pct(kill['median_unc_gap_reduction'])}  (>=25%? {kill['crit_a_median_ge25pct']})"
    )
    print(
        f"  (b) fraction of {kill['n_certified_opening']} certified closing <=10 nodes = "
        f"{fmt_pct(kill['frac_certified_closed_10'])}  (>=30%? {kill['crit_b_frac_ge30pct']})"
    )
    print(f"\n  ==> PHASE-2 PREMISE {'SURVIVES' if kill['premise_survives'] else 'FALSIFIED'}")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="T2.1 root-loop replay entry experiment")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument(
        "--instances",
        default=None,
        help="comma-separated subset of the panel (default: full panel)",
    )
    ap.add_argument("--harvest-s", type=float, default=8.0)
    ap.add_argument("--loop-budget-s", type=float, default=30.0)
    ap.add_argument("--obbt-stage-s", type=float, default=10.0)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument(
        "--per-instance-cap-s",
        type=float,
        default=240.0,
        help="skip an instance whose harvest already blew past this (soft)",
    )
    ap.add_argument("--no-oracle-variant", action="store_true")
    ap.add_argument(
        "--jsonl-out",
        default=None,
        help="append one JSON record per completed instance (survives a kill)",
    )
    args = ap.parse_args()

    panel = PANEL if args.instances is None else [s.strip() for s in args.instances.split(",")]

    results: list[dict] = []
    ran: list[str] = []
    skipped: list[str] = []
    p0_hit = False

    def _emit(rec: dict) -> None:
        if not args.jsonl_out:
            return
        import json

        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items() if not isinstance(v, np.ndarray)}
            if isinstance(o, (list, tuple)):
                return [_clean(x) for x in o]
            if isinstance(o, np.ndarray):
                return None
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            return o

        with open(args.jsonl_out, "a") as fh:
            fh.write(json.dumps(_clean(rec)) + "\n")
            fh.flush()

    for name in panel:
        path = resolve_nl(name, args.snapshot)
        if path is None:
            print(f"[{name}] NL not found in snapshot/corpus -> SKIP")
            skipped.append(f"{name}(no-nl)")
            continue
        print(f"\n########## {name} ##########")
        t0 = time.perf_counter()
        try:
            r = replay_instance(
                name,
                path,
                harvest_s=args.harvest_s,
                loop_budget_s=args.loop_budget_s,
                obbt_stage_s=args.obbt_stage_s,
                resolve_s=60.0,
                max_iters=args.max_iters,
                oracle_variant=not args.no_oracle_variant,
            )
            wall = time.perf_counter() - t0
            r["total_wall"] = wall
            results.append(r)
            ran.append(name)
            _emit(r)
            loop = r.get("loop", {})
            tu = r.get("tree_untightened", {})
            ti = r.get("tree_tightened", {})
            print(
                f"[{name}] done in {wall:.1f}s  base_gap="
                f"{r.get('baseline_gap')}  final_gap={loop.get('final_gap')}  "
                f"gap_red={fmt_pct(loop.get('gap_reduction'))}  "
                f"nodes_un={tu.get('nodes')} nodes_ti={ti.get('nodes')} "
                f"closed10={r.get('closed_within_10_by_reduction')}"
            )
        except P0StopError as e:
            p0_hit = True
            print("\n" + "!" * 78)
            print("P0 SOUNDNESS VIOLATION — STOP (§0.6)")
            print("!" * 78)
            print(str(e))
            print("!" * 78)
            # Record and STOP the whole run: a P0 must be surfaced, not buried.
            break
        except Exception as e:
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            skipped.append(f"{name}(error)")
            continue

    if p0_hit:
        print("\nABORTED on P0 soundness violation. No kill verdict computed.")
        return

    kill = evaluate_kill(results)
    print_summary(results, kill, ran, skipped)


if __name__ == "__main__":
    main()
