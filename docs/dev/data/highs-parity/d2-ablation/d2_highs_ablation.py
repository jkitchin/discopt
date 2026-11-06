#!/usr/bin/env python
"""D2 HiGHS ablation: which HiGHS MIP components certify the instances Rust fails.

For every (instance, arm) a FRESH process runs raw highspy on the arrays produced by
``lp_milp_loader.read_instance`` (the same arrays discopt's routes see), with one HiGHS
component disabled through its documented option (names from
``highs/lp_data/HighsOptions.h`` of HiGHS 1.15.1; the mip code's read sites were checked
in ``mip/HighsMipSolverData.cpp`` / ``mip/HighsMipSolver.cpp``). The HiGHS log is parsed
for the root bound before/after cuts, the solution-source letters of every improving
incumbent, restarts, cut counts and the LP-iteration split.

Discipline (CLAUDE.md §6-§10): executed-solve counter printed at the end and non-zero exit
on 0; no exception swallowed (a worker crash is recorded as an ``error`` row AND counted
separately, never as a result); HiGHS version recorded per row; load average recorded
before and after each solve; progress unbuffered per solve.

Usage:
  python d2_highs_ablation.py run --out X.jsonl [--arms a,b] [--instances n1,n2] [--jobs 1]
  python d2_highs_ablation.py worker <path> <arm> <time_limit> <gap> <oracle> <sense>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/home/user/discopt")
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))

HERE = Path(__file__).resolve()
GAP = 1e-4
TIME_LIMIT = 20.0

# HiGHS 1.15.1 lp_data/HConst.h:260-288 PresolveRuleType (bit index = enum value).
PRESOLVE_RULE = {
    "ForcingRow": 6, "ForcingCol": 7, "FreeColSubstitution": 8, "DoubletonEquation": 9,
    "DependentEquations": 10, "DependentFreeCols": 11, "Aggregator": 12,
    "ParallelRowsAndCols": 13, "Sparsify": 14, "Probing": 15, "Enumeration": 16,
    "DualFixing": 17, "ColStuffing": 18, "InitialSweep": 19,
}

# Every arm except ``base_tdef`` runs with threads=1.
ARMS: dict[str, dict] = {
    "base_tdef": {},
    "base": {},
    "presolve_off": {"presolve": "off"},
    "restart_off": {"mip_allow_restart": False},
    "symmetry_off": {"mip_detect_symmetry": False},
    "heur_all_off": {
        "mip_heuristic_effort": 0.0,
        "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False,
        "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
    },
    "heur_effort_0": {"mip_heuristic_effort": 0.0},
    "fj_off": {"mip_heuristic_run_feasibility_jump": False},
    "rins_off": {"mip_heuristic_run_rins": False},
    "rens_off": {"mip_heuristic_run_rens": False},
    "rootredcost_off": {"mip_heuristic_run_root_reduced_cost": False},
    "node_cuts_off": {"mip_allow_cut_separation_at_nodes": False},
    # No option switches root separation off; this is the tightest throttle the options
    # allow: tableau separator refuses when the pool holds > soft_limit cuts
    # (HighsTableauSeparator.cpp:47), LP rows age out after 0 non-binding epochs
    # (HighsLpRelaxation.cpp:640-663), pool rows age out immediately (HighsCutPool.cpp:
    # 162-220). Conflict pools share the soft limit (HighsMipSolver.cpp:290).
    "cuts_throttle": {"mip_pool_soft_limit": 1, "mip_lp_age_limit": 0, "mip_pool_age_limit": 0},
    "pool_soft_1": {"mip_pool_soft_limit": 1},
    "lp_age_0": {"mip_lp_age_limit": 0, "mip_pool_age_limit": 0},
    # minreliable=0: every column counts as reliable (HighsPseudocost.h:243-245), so the
    # reliability-branching strong-branch probes never run.
    "sb_off": {"mip_pscost_minreliable": 0},
    "sb_heavy": {"mip_pscost_minreliable": 100000},
    "probing_off": {"presolve_rule_off": 1 << PRESOLVE_RULE["Probing"]},
    "parallel_rc_off": {"presolve_rule_off": 1 << PRESOLVE_RULE["ParallelRowsAndCols"]},
    "aggregator_off": {"presolve_rule_off": 1 << PRESOLVE_RULE["Aggregator"]},
    "doubleton_off": {"presolve_rule_off": 1 << PRESOLVE_RULE["DoubletonEquation"]},
    "sparsify_off": {"presolve_rule_off": 1 << PRESOLVE_RULE["Sparsify"]},
    "dualfix_off": {"presolve_rule_off": 1 << PRESOLVE_RULE["DualFixing"]},
    "presolve_reduce_only": {
        # keep the reductions, drop every "MIP-aware" rule (probing, dual fixing, parallel,
        # enumeration, lifting) -- what a pure LP presolve would do
        "presolve_rule_off": sum(
            1 << PRESOLVE_RULE[k] for k in ("Probing", "DualFixing", "ParallelRowsAndCols",
                                             "Enumeration", "ColStuffing")
        ),
    },
    "restart_off+heur_all_off": {
        "mip_allow_restart": False,
        "mip_heuristic_effort": 0.0,
        "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False,
        "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
    },
    # ---- phase 2: leave-one-IN (which single component suffices on top of plain B&B?)
    "all_off": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_allow_cut_separation_at_nodes": False,
        "mip_pool_soft_limit": 1, "mip_lp_age_limit": 0, "mip_pool_age_limit": 0,
        "mip_pscost_minreliable": 0,
    },
    "keep_cuts": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_pscost_minreliable": 0,
    },
    "keep_heur": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_allow_cut_separation_at_nodes": False,
        "mip_pool_soft_limit": 1, "mip_lp_age_limit": 0, "mip_pool_age_limit": 0,
        "mip_pscost_minreliable": 0,
    },
    "keep_presolve": {
        "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_allow_cut_separation_at_nodes": False,
        "mip_pool_soft_limit": 1, "mip_lp_age_limit": 0, "mip_pool_age_limit": 0,
        "mip_pscost_minreliable": 0,
    },
    "keep_sb": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_allow_cut_separation_at_nodes": False,
        "mip_pool_soft_limit": 1, "mip_lp_age_limit": 0, "mip_pool_age_limit": 0,
    },
    "keep_cuts_heur": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_pscost_minreliable": 0,
    },
    "keep_cuts_presolve": {
        "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_pscost_minreliable": 0,
    },
    "keep_cuts_sb": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
    },
    # ---- phase 4 (Phase B entry B0, 2026-11-06): which part of `keep_cuts` certifies?
    # `keep_cuts` minus node separation; minus aging (LP age at its maximum, pool age
    # at its maximum, pool soft limit at its maximum, so rows are effectively never
    # removed).
    "keep_cuts_no_nodesep": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_pscost_minreliable": 0,
        "mip_allow_cut_separation_at_nodes": False,
    },
    "keep_cuts_no_aging": {
        "presolve": "off", "mip_allow_restart": False, "mip_detect_symmetry": False,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
        "mip_pscost_minreliable": 0,
        "mip_lp_age_limit": 32767, "mip_pool_age_limit": 1000,
        "mip_pool_soft_limit": 2147483647,
    },
    # ---- phase 2: root-only (mip_max_nodes=1) bound/incumbent decomposition
    "root_only": {"mip_max_nodes": 1},
    "root_only+cuts_throttle": {"mip_max_nodes": 1, "mip_pool_soft_limit": 1,
                                "mip_lp_age_limit": 0, "mip_pool_age_limit": 0},
    "root_only+presolve_off": {"mip_max_nodes": 1, "presolve": "off"},
    "root_only+restart_off": {"mip_max_nodes": 1, "mip_allow_restart": False},
    "root_only+heur_all_off": {
        "mip_max_nodes": 1,
        "mip_heuristic_effort": 0.0, "mip_heuristic_run_feasibility_jump": False,
        "mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False,
        "mip_heuristic_run_root_reduced_cost": False,
    },
    "restart_off+cuts_throttle": {
        "mip_allow_restart": False,
        "mip_pool_soft_limit": 1, "mip_lp_age_limit": 0, "mip_pool_age_limit": 0,
    },
}
PHASE1 = [
    "base_tdef", "base", "presolve_off", "restart_off", "symmetry_off", "heur_all_off",
    "node_cuts_off", "cuts_throttle", "sb_off", "probing_off",
]

# Rust fails / HiGHS certifies (p6.6 miplib.json + lmi.json panel M), computed in-session.
MIPLIB_FAIL = [
    "neos-1425699", "mik-250-20-75-5", "neos-911970", "mik-250-20-75-3", "fiber", "khb05250",
    "nexp-50-20-1-1", "mik-250-20-75-2", "nexp-50-20-4-2", "gt2", "mik-250-20-75-1",
    "sp150x300d", "neos17", "beavma", "neos-3611447-jijia", "neos-3610051-istra",
    "neos-3611689-kaihu", "neos-3610040-iskar", "b-ball", "bppc8-02", "gsvm2rl3",
]
M_ONLY_FAIL = ["2122", "gesa2", "issue-2446", "p0548"]


def load_instances() -> list[dict]:
    pm = json.loads((REPO / "discopt_benchmarks/data/miplib/panel.json").read_text())
    mm = json.loads((REPO / "discopt_benchmarks/data/lp_milp/manifest.json").read_text())
    out = []
    by = {x["name"]: x for x in pm["members"]}
    for n in MIPLIB_FAIL:
        x = by[n]
        out.append({"name": n, "panel": "MIPLIB", "path": str(REPO / x["path"]),
                    "oracle": x["oracle_objective"]})
    by = {x["name"]: x for x in mm["members"] if x.get("panel") == "M"}
    for n in M_ONLY_FAIL:
        x = by[n]
        out.append({"name": n, "panel": "M", "path": str(REPO / x["path"]),
                    "oracle": x["oracle_objective"]})
    return out


def _tol(ref: float, gap: float) -> float:
    return 1e-6 * (1.0 + abs(ref)) + gap * abs(ref)


# ----------------------------------------------------------------------------- log parsing
_DISP = re.compile(
    r"^ ?(?P<src>[A-Za-z ])\s+(?P<proc>\d+)\s+(?P<inq>\d+)\s+(?P<leaves>\d+)\s+"
    r"(?P<expl>[\d.]+)%\s+(?P<bound>\S+)\s+(?P<sol>\S+)\s+(?P<gap>\S+)\s+"
    r"(?P<cuts>\d+)\s+(?P<inlp>\d+)\s+(?P<confl>\d+)\s+(?P<iters>\d+)\s+(?P<time>[\d.]+)s$"
)


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def parse_log(text: str) -> dict:
    lines = text.splitlines()
    disp = []
    for ln in lines:
        m = _DISP.match(ln)
        if m:
            d = m.groupdict()
            disp.append({
                "src": d["src"].strip() or None, "proc": int(d["proc"]), "inq": int(d["inq"]),
                "leaves": int(d["leaves"]), "bound": _num(d["bound"]), "sol": _num(d["sol"]),
                "cuts": int(d["cuts"]), "inlp": int(d["inlp"]), "confl": int(d["confl"]),
                "iters": int(d["iters"]), "t": float(d["time"]),
            })
    out: dict = {"n_display": len(disp)}
    out["sol_sources"] = [
        {"src": d["src"], "sol": d["sol"], "proc": d["proc"], "t": d["t"]} for d in disp if d["src"]
    ]
    import math

    finite = [d for d in disp if d["bound"] is not None and math.isfinite(d["bound"])]
    out["root_lp_bound"] = finite[0]["bound"] if finite else None
    root = [d for d in finite if d["proc"] == 0]
    out["root_end_bound"] = root[-1]["bound"] if root else None
    out["root_end_cuts"] = root[-1]["cuts"] if root else None
    out["root_end_inlp"] = root[-1]["inlp"] if root else None
    out["root_end_t"] = root[-1]["t"] if root else None
    out["root_end_sol"] = root[-1]["sol"] if root else None
    out["last_cuts"] = disp[-1]["cuts"] if disp else None
    out["last_inlp"] = disp[-1]["inlp"] if disp else None
    out["last_confl"] = disp[-1]["confl"] if disp else None
    out["restarts"] = sum("inactive integer columns, restarting" in ln for ln in lines)
    out["symmetry"] = next((ln.strip() for ln in lines if "ymmetry" in ln), None)
    out["presolve_reductions"] = [ln.strip() for ln in lines if ln.startswith("Presolve reductions")]
    out["presolve_rules_off"] = [ln.strip() for ln in lines if ln.strip().startswith("Rule ")]
    out["model_after_restart"] = [ln.strip() for ln in lines if "Model after restart" in ln]
    out["presolved_dims"] = next(
        (ln.strip() for ln in lines if re.match(r"^\d+ rows, \d+ cols, \d+ nonzeros", ln.strip())),
        None,
    )
    rep: dict = {}
    for ln in lines:
        s = ln.strip()
        for key in ("Nodes", "Repair LPs", "LP iterations", "Max sub-MIP depth", "Timing"):
            if s.startswith(key + " "):
                rep[key] = _num(s.split()[-1] if key != "LP iterations" else s.split()[2])
        for key in ("strong br.", "separation", "heuristics"):
            if s.endswith(f"({key})"):
                rep["iters_" + key.replace(" ", "_").rstrip(".")] = _num(s.split()[0])
    out["report"] = rep
    return out


# ----------------------------------------------------------------------------- worker
def worker(path: str, arm: str, time_limit: float, gap: float, oracle: float) -> dict:
    import highspy
    import lp_milp_loader as L  # noqa: N812

    inst = L.read_instance(path)
    h = highspy.Highs()
    log_path = os.path.join(tempfile.gettempdir(), f"d2abl_{os.getpid()}.log")
    opts: list[tuple] = [
        ("output_flag", True), ("log_to_console", False), ("log_file", log_path),
        ("time_limit", float(time_limit)), ("mip_rel_gap", float(gap)),
    ]
    if arm != "base_tdef":
        opts.append(("threads", 1))
    opts += list(ARMS[arm].items())
    for k, v in opts:
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {k}={v!r}")
    # read back every option we set, so a silently-ignored option is caught
    readback = {}
    for k, v in opts:
        if k in ("output_flag", "log_to_console", "log_file"):
            continue
        rb = h.getOptionValue(k)
        got = rb[1] if isinstance(rb, tuple) else rb
        readback[k] = got
        if isinstance(v, bool):
            ok = bool(got) == v
        elif isinstance(v, (int, float)):
            ok = abs(float(got) - float(v)) <= 1e-12
        else:
            ok = str(got) == str(v)
        if not ok:
            raise RuntimeError(f"option {k} read back as {got!r}, set {v!r}")
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = inst.n, inst.m
    lp.col_cost_ = inst.c
    lp.offset_ = inst.offset
    lp.sense_ = highspy.ObjSense.kMaximize if inst.maximize else highspy.ObjSense.kMinimize
    lp.col_lower_, lp.col_upper_ = inst.col_lo, inst.col_hi
    lp.row_lower_, lp.row_upper_ = inst.row_lo, inst.row_hi
    a_csc = inst.A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = a_csc.indptr
    lp.a_matrix_.index_ = a_csc.indices
    lp.a_matrix_.value_ = a_csc.data
    if not inst.is_int.any():
        raise RuntimeError(f"{inst.name}: no integer columns -- not a MILP")
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
        for b in inst.is_int
    ]
    st = h.passModel(lp)
    if st != highspy.HighsStatus.kOk:
        raise RuntimeError(f"{inst.name}: passModel returned {st}")
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    st = h.run()
    wall = time.perf_counter() - t0
    load1 = os.getloadavg()
    log_text = Path(log_path).read_text()
    os.remove(log_path)
    if st == highspy.HighsStatus.kError:
        raise RuntimeError(f"{inst.name}/{arm}: run returned kError; log tail:\n{log_text[-2000:]}")
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    import numpy as np

    x = np.asarray(h.getSolution().col_value, float)
    has_x = x.shape[0] == inst.n and info.primal_solution_status > 0
    obj = float(info.objective_function_value) if has_x else None
    bound = float(info.mip_dual_bound)
    max_viol = L.dense_max_violation(inst, x) if has_x else None
    certified = status == "Optimal"
    tol = _tol(oracle, gap)
    incorrect: list[str] = []
    if certified and (obj is None or abs(obj - oracle) > tol):
        incorrect.append(f"certified objective {obj} vs oracle {oracle} (tol {tol:.3g})")
    if inst.maximize:
        if bound < oracle - tol:
            incorrect.append(f"dual bound {bound} below oracle {oracle} (max)")
    else:
        if bound > oracle + tol:
            incorrect.append(f"dual bound {bound} above oracle {oracle} (min)")
    if has_x and max_viol is not None and max_viol > 1e-5:
        incorrect.append(f"incumbent violation {max_viol:.3g}")
    primal_gap = None if obj is None else abs(obj - oracle) / max(1.0, abs(oracle))
    dual_gap = abs(bound - oracle) / max(1.0, abs(oracle)) if np.isfinite(bound) else None
    parsed = parse_log(log_text)
    return {
        "name": inst.name, "arm": arm, "options": {k: v for k, v in opts if k not in (
            "output_flag", "log_to_console", "log_file")},
        "options_readback": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                             for k, v in readback.items()},
        "highs_version": h.version(), "status": status, "certified": certified,
        "objective": obj, "bound": bound, "mip_gap": float(info.mip_gap),
        "primal_gap": primal_gap, "dual_gap": dual_gap,
        "nodes": int(info.mip_node_count), "lp_iters": int(info.simplex_iteration_count),
        "wall": wall, "max_viol": max_viol, "incorrect": incorrect, "oracle": oracle,
        "n": inst.n, "m": inst.m, "n_int": int(inst.is_int.sum()),
        "load_before": load0, "load_after": load1, "log": parsed,
        "log_tail": log_text[-1200:],
    }


# ----------------------------------------------------------------------------- driver
def _run_sub(inst: dict, arm: str, time_limit: float, gap: float) -> dict:
    cmd = [sys.executable, "-I", str(HERE), "worker", inst["path"], arm, str(time_limit),
           str(gap), repr(inst["oracle"])]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * time_limit + 180)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"name": inst["name"], "arm": arm, "error": f"rc={p.returncode}",
                "stderr_tail": p.stderr[-3000:], "stdout_tail": p.stdout[-1000:]}
    return json.loads(lines[0][len("RESULT "):])


def cmd_run(a) -> int:
    import highspy

    insts = load_instances()
    if a.instances:
        want = set(a.instances.split(","))
        missing = want - {i["name"] for i in insts}
        if missing:
            raise SystemExit(f"unknown instances {sorted(missing)}")
        insts = [i for i in insts if i["name"] in want]
    arms = a.arms.split(",") if a.arms else PHASE1
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; known {sorted(ARMS)}")
    out = Path(a.out)
    done = set()
    if a.resume and out.exists():
        for ln in out.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if "error" not in r:
                    done.add((r["name"], r["arm"]))
    jobs = [(i, arm) for i in insts for arm in arms if (i["name"], arm) not in done]
    print(f"highspy {highspy.Highs().version()} | {len(insts)} instances x {len(arms)} arms = "
          f"{len(jobs)} solves ({len(done)} already done) | jobs={a.jobs} | "
          f"load {os.getloadavg()}", flush=True)
    executed = 0
    errors = 0
    certified = 0
    t_start = time.perf_counter()
    with out.open("a") as f, ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_run_sub, i, arm, a.time_limit, a.gap): (i, arm) for i, arm in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            r["name"] = futs[fut][0]["name"]
            r["panel"] = futs[fut][0]["panel"]
            r["t_submit"] = time.time()
            f.write(json.dumps(r) + "\n")
            f.flush()
            if "error" in r:
                errors += 1
                print(f"[{k}/{len(jobs)}] {r['name']:20s} {r['arm']:18s} ERROR {r['error']} "
                      f"{r.get('stderr_tail', '')[-300:]}", flush=True)
                continue
            executed += 1
            certified += r["certified"]
            lg = r["log"]
            srcs = "".join(s["src"] for s in lg["sol_sources"])
            print(
                f"[{k}/{len(jobs)}] {r['name']:20s} {r['arm']:18s} {r['status']:12s} "
                f"n={r['nodes']:7d} w={r['wall']:5.1f}s pg={r['primal_gap'] if r['primal_gap'] is None else round(r['primal_gap'], 5)} "
                f"dg={None if r['dual_gap'] is None else round(r['dual_gap'], 5)} "
                f"root={lg['root_lp_bound']}->{lg['root_end_bound']} cuts={lg['root_end_cuts']} "
                f"rst={lg['restarts']} src={srcs} load={r['load_before'][0]:.1f} "
                f"{'INCORRECT ' + str(r['incorrect']) if r['incorrect'] else ''}",
                flush=True,
            )
    print(f"executed solves: {executed}; worker errors: {errors}; certified: {certified}; "
          f"wall {time.perf_counter() - t_start:.0f}s", flush=True)
    if executed == 0:
        print("NO SOLVE EXECUTED", flush=True)
        return 2
    return 0


def main(argv=None) -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _, _, path, arm, tl, gap, oracle = sys.argv
        res = worker(path, arm, float(tl), float(gap), float(oracle))
        print("RESULT " + json.dumps(res), flush=True)
        return 0
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--arms", default="")
    r.add_argument("--instances", default="")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    r.add_argument("--gap", type=float, default=GAP)
    r.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
