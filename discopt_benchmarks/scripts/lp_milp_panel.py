"""LP/MILP routing panel: R (Rust route) vs H (HiGHS route) vs H0 (raw highspy).

Stage 0 of ``docs/dev/lp-milp-highs-routing-plan.md`` (§6). Subcommands:

``screen``  build a panel manifest: read each instance, validate the loader (raw HiGHS on
            the original file = oracle; raw HiGHS on the converted arrays must reproduce
            it; the ``.nl`` round-trip must load with the right shape). Mismatches are
            *excluded and listed*, never dropped silently.
``run``     solve manifest instances with the requested arms, rounds interleaved per
            instance. R/H run ``Model.solve`` in a subprocess with
            ``DISCOPT_LP_MILP_BACKEND=rust|highs``; H0 runs raw highspy in-process.
``tiny``    E1: per-call overhead on generated 2-20 variable LP/MILPs, in-process.

Discipline (CLAUDE.md §6-§10): every run ends with an executed-comparison count and
exits non-zero when it is 0; exceptions in the instrument propagate; the loaded
``discopt`` is asserted to be the one under test; progress is unbuffered per item.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import lp_milp_loader as L  # noqa: E402, N812

GAP = 1e-4


# ----------------------------------------------------------------------------- helpers
def _header() -> dict:
    import highspy

    commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    h = highspy.Highs()
    return {
        "load": os.getloadavg(),
        "commit": commit,
        "highspy": getattr(highspy, "__version__", None) or h.version(),
        "highs_version": h.version(),
        "python": sys.executable,
    }


def _assert_discopt_under_test():
    import discopt

    want = str(REPO / "python")
    if not discopt.__file__.startswith(want):
        raise RuntimeError(f"loaded discopt from {discopt.__file__}, expected under {want}")
    return discopt


def _tol(ref: float, gap: float) -> float:
    return 1e-6 * (1.0 + abs(ref)) + gap * abs(ref)


def solve_arrays(inst: L.LinearInstance, time_limit: float, gap: float = GAP) -> dict:
    """H0: raw highspy on the arrays (default threads, as the route; matched relative gap)."""
    import highspy

    h = highspy.Highs()
    log_path = os.path.join(
        tempfile.gettempdir(), f"lp_milp_h0_{os.getpid()}_{time.perf_counter_ns()}.log"
    )
    opts = [
        ("output_flag", True),
        ("log_to_console", False),
        ("log_file", log_path),
        ("time_limit", float(time_limit)),
        ("mip_rel_gap", float(gap)),
    ]
    for k, v in opts:
        if h.setOptionValue(k, v) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected option {k}={v}")
    t0 = time.perf_counter()
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
    if inst.is_int.any():
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
            for b in inst.is_int
        ]
    st = h.passModel(lp)
    if st != highspy.HighsStatus.kOk:
        raise RuntimeError(f"{inst.name}: passModel returned {st}; log {log_path}")
    st = h.run()
    if st == highspy.HighsStatus.kError:
        with open(log_path) as f:
            tail = f.read()[-2000:]
        raise RuntimeError(
            f"{inst.name}: run returned {st}, model status "
            f"{h.modelStatusToString(h.getModelStatus())}; HiGHS log:\n{tail}"
        )
    os.remove(log_path)
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    x = np.asarray(h.getSolution().col_value, float)
    wall = time.perf_counter() - t0
    has_x = x.shape[0] == inst.n and info.primal_solution_status > 0
    is_mip = bool(inst.is_int.any())
    obj = float(info.objective_function_value) if has_x else None
    return {
        "status": status,
        "objective": obj,
        "bound": float(info.mip_dual_bound) if is_mip else obj,
        "nodes": int(info.mip_node_count) if is_mip else 0,
        "wall": wall,
        "x": x if has_x else None,
    }


# ----------------------------------------------------------------------------- screen
def cmd_screen(a) -> int:
    from discopt.modeling.core import from_nl

    _assert_discopt_under_test()
    print("header", json.dumps(_header()), flush=True)
    files = []
    for s in a.sources:
        p = Path(s)
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix in (".mps", ".lp"))
        else:
            files.append(p)
    members, excluded = [], []
    compared = 0
    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(files, 1):
            tag = f"[screen {i}/{len(files)}] {f.name:28s}"
            try:
                inst = L.read_instance(str(f))
            except (ValueError, RuntimeError) as e:
                # The loader's explicit refusals (unsupported feature, unreadable file):
                # recorded with the reason, not swallowed.
                excluded.append({"file": str(f), "reason": f"loader: {e}"})
                print(tag, "EXCLUDE loader:", e, flush=True)
                continue
            n_int = int(inst.is_int.sum())
            nnz = int(inst.A.nnz)
            if a.milp_only and n_int == 0:
                excluded.append({"file": str(f), "reason": "no integer columns"})
                print(tag, "EXCLUDE no integers", flush=True)
                continue
            if nnz > a.max_nnz:
                excluded.append({"file": str(f), "reason": f"nnz {nnz} > {a.max_nnz}"})
                print(tag, f"EXCLUDE nnz={nnz}", flush=True)
                continue
            ref = L.highs_reference(str(f), time_limit=a.time_limit)
            if ref["status"] != "Optimal":
                excluded.append({"file": str(f), "reason": f"oracle status {ref['status']}"})
                print(tag, "EXCLUDE oracle", ref["status"], flush=True)
                continue
            h0 = solve_arrays(inst, a.time_limit)
            compared += 1
            if h0["status"] != "Optimal" or abs(h0["objective"] - ref["objective"]) > _tol(
                ref["objective"], 0.0 if n_int == 0 else GAP
            ):
                excluded.append(
                    {
                        "file": str(f),
                        "reason": f"array mismatch: {h0['status']} {h0['objective']} "
                        f"vs oracle {ref['objective']}",
                    }
                )
                print(tag, "EXCLUDE array mismatch", h0["objective"], ref["objective"], flush=True)
                continue
            nl = os.path.join(td, f"{inst.name}.nl")
            try:
                L.write_nl(inst, nl)
            except ValueError as e:
                excluded.append({"file": str(f), "reason": f"nl writer: {e}"})
                print(tag, "EXCLUDE nl writer:", e, flush=True)
                continue
            mdl = from_nl(nl)
            nv = sum(v.size for v in mdl._variables)
            if nv != inst.n:
                raise RuntimeError(f"{inst.name}: .nl round-trip has {nv} vars, expected {inst.n}")
            viol = L.dense_max_violation(inst, h0["x"])
            members.append(
                {
                    "name": inst.name,
                    "path": os.path.relpath(f, REPO),
                    "panel": a.panel,
                    "n": inst.n,
                    "m": inst.m,
                    "nnz": nnz,
                    "n_int": n_int,
                    "oracle_objective": ref["objective"],
                    "oracle_nodes": ref["nodes"],
                    "oracle_max_viol": viol,
                }
            )
            print(tag, f"OK n={inst.n} m={inst.m} nnz={nnz} int={n_int} obj={ref['objective']}",
                  flush=True)  # fmt: skip
    out = Path(a.out)
    old = json.loads(out.read_text()) if out.exists() else {"members": [], "excluded": []}
    keep = [x for x in old["members"] if x["panel"] != a.panel]
    keep_ex = [x for x in old["excluded"] if x.get("panel") != a.panel]
    for x in excluded:
        x["panel"] = a.panel
    out.write_text(
        json.dumps({"members": keep + members, "excluded": keep_ex + excluded}, indent=1) + "\n"
    )
    print(f"screen: panel={a.panel} members={len(members)} excluded={len(excluded)} "
          f"compared={compared}", flush=True)  # fmt: skip
    return 0 if compared > 0 else 1


# ----------------------------------------------------------------------------- run
def worker(path: str, arm: str, time_limit: float, gap: float, max_nodes: int) -> dict:
    import discopt.solver as solver_mod
    from discopt.modeling.core import from_nl

    _assert_discopt_under_test()
    if arm == "H" and not hasattr(solver_mod, "_solve_milp_highs"):
        return {"skipped": "H arm not yet available (discopt.solver has no _solve_milp_highs)"}
    inst = L.read_instance(path)
    with tempfile.TemporaryDirectory() as td:
        nl = os.path.join(td, f"{inst.name}.nl")
        perm = L.write_nl(inst, nl)
        mdl = from_nl(nl)
    # from_nl yields scalar variables x0..x{n-1} in .nl column order; .nl position k holds
    # original column perm[k].
    names = [v.name for v in mdl._variables]
    if names != [f"x{k}" for k in range(inst.n)]:
        raise RuntimeError(f"{inst.name}: unexpected from_nl variable layout {names[:5]}...")
    t0 = time.perf_counter()
    r = mdl.solve(time_limit=time_limit, gap_tolerance=gap, max_nodes=max_nodes)
    wall = time.perf_counter() - t0
    viol = None
    if r.x is not None:
        x = np.empty(inst.n)
        for k in range(inst.n):
            x[perm[k]] = float(np.asarray(r.x[f"x{k}"], float).ravel()[0])
        viol = L.dense_max_violation(inst, x)
    stats = r.solver_stats or {}
    return {
        "status": r.status,
        "objective": r.objective,
        "bound": r.bound,
        "gap_certified": bool(r.gap_certified),
        "nodes": int(r.node_count),
        "wall": wall,
        "route": stats.get("route/lp_milp_backend"),
        "max_viol": viol,
        "env_backend": os.environ.get("DISCOPT_LP_MILP_BACKEND"),
    }


def _run_sub(path: str, arm: str, a) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env["DISCOPT_LP_MILP_BACKEND"] = {"R": "rust", "H": "highs"}[arm]
    cmd = [sys.executable, str(HERE), "worker", path, arm, str(a.time_limit), str(a.gap),
           str(a.max_nodes)]  # fmt: skip
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3 * a.time_limit + 120)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"error": f"rc={p.returncode}", "stderr_tail": p.stderr[-1500:]}
    return json.loads(lines[0][len("RESULT ") :])


def _incorrect(res: dict, oracle: float, maximize: bool, gap: float) -> list[str]:
    why = []
    tol = _tol(oracle, gap)
    st = res.get("status")
    obj = res.get("objective")
    if st in ("optimal", "Optimal") and (obj is None or abs(obj - oracle) > tol):
        why.append(f"optimal objective {res.get('objective')} vs oracle {oracle}")
    if st in ("infeasible", "unbounded", "Infeasible", "Unbounded"):
        why.append(f"status {st} on an instance with oracle optimum {oracle}")
    b = res.get("bound")
    crosses = (
        b is not None
        and np.isfinite(b)
        and ((not maximize and b > oracle + tol) or (maximize and b < oracle - tol))
    )
    if crosses:
        why.append(f"bound {b} crosses oracle {oracle}")
    mv = res.get("max_viol")
    if st in ("optimal", "Optimal", "feasible") and mv is not None and mv > 1e-5:
        why.append(f"returned point violates by {mv:.3g}")
    return why


def cmd_run(a) -> int:
    _assert_discopt_under_test()
    man = json.loads(Path(a.manifest).read_text())
    mem = [x for x in man["members"] if (not a.panel or x["panel"] in a.panel)]
    if a.instances:
        want = set(a.instances.split(","))
        mem = [x for x in mem if x["name"] in want]
        missing = want - {x["name"] for x in mem}
        if missing:
            raise SystemExit(f"instances not in manifest: {sorted(missing)}")
    arms = a.arms.split(",")
    hdr = _header()
    print("header", json.dumps(hdr), "arms", arms, "n", len(mem), flush=True)
    rows, executed, incorrect = [], 0, 0
    for i, x in enumerate(mem, 1):
        path = str(REPO / x["path"])
        inst = L.read_instance(path)
        per = {arm: [] for arm in arms}
        for rnd in range(a.rounds):
            for arm in arms if rnd % 2 == 0 else arms[::-1]:
                if arm == "H0":
                    r = solve_arrays(inst, a.time_limit, a.gap)
                    x_h0 = r.pop("x")
                    r["max_viol"] = None if x_h0 is None else L.dense_max_violation(inst, x_h0)
                    r["status"] = {"Optimal": "optimal"}.get(r["status"], r["status"])
                else:
                    r = _run_sub(path, arm, a)
                if "skipped" in r or "error" in r:
                    per[arm].append(r)
                    continue
                r["incorrect"] = _incorrect(r, x["oracle_objective"], inst.maximize, a.gap)
                executed += 1
                incorrect += bool(r["incorrect"])
                per[arm].append(r)
        summ = {"name": x["name"], "panel": x["panel"], "oracle": x["oracle_objective"]}
        for arm, rs in per.items():
            ok = [r for r in rs if "skipped" not in r and "error" not in r]
            if not ok:
                summ[arm] = rs[0]
                continue
            walls = [r["wall"] for r in ok]
            last = ok[-1]
            summ[arm] = {
                **{k: last.get(k) for k in ("status", "objective", "bound", "gap_certified",
                                            "nodes", "route", "max_viol")},
                "wall_med": statistics.median(walls),
                "wall_sd": statistics.stdev(walls) if len(walls) > 1 else 0.0,
                "nodes_all": [r.get("nodes") for r in ok],
                "incorrect": sorted({w for r in ok for w in r["incorrect"]}),
            }  # fmt: skip
        rows.append(summ)
        brief = {arm: (v.get("status"), v.get("nodes"), v.get("objective"),
                       round(v["wall_med"], 3) if "wall_med" in v else v)
                 for arm, v in summ.items() if arm in arms}  # fmt: skip
        print(f"[run {i}/{len(mem)}] {x['name']:24s} {brief}", flush=True)
        for arm in arms:
            if summ[arm].get("incorrect"):
                print(f"  !!! INCORRECT {arm}: {summ[arm]['incorrect']}", flush=True)
    hdr["load_end"] = os.getloadavg()
    out = {"header": hdr, "args": vars(a), "rows": rows, "executed": executed,
           "incorrect_count": incorrect}  # fmt: skip
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"run: executed={executed} incorrect_count={incorrect} load_end={hdr['load_end']}",
          flush=True)  # fmt: skip
    return 0 if executed > 0 else 1


# ----------------------------------------------------------------------------- tiny
def tiny_instances(count: int, seed: int) -> list[L.LinearInstance]:
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    out = []
    for k in range(count):
        n = int(rng.integers(2, 21))
        m = int(rng.integers(1, n + 1))
        a_dense = rng.integers(-5, 6, size=(m, n)).astype(float)
        a_dense[a_dense == 0] = 1.0
        lo, hi = np.zeros(n), np.full(n, 10.0)
        is_int = (rng.random(n) < 0.5) if k % 2 else np.zeros(n, dtype=bool)
        x0 = rng.integers(0, 11, size=n).astype(float)  # integral, inside the box
        slack = rng.integers(0, 6, size=m).astype(float)
        out.append(
            L.LinearInstance(
                name=f"tiny{k:03d}",
                c=rng.integers(-9, 10, size=n).astype(float),
                offset=0.0,
                maximize=False,
                A=sp.csr_matrix(a_dense),
                row_lo=np.full(m, -np.inf),
                row_hi=a_dense @ x0 + slack,
                col_lo=lo,
                col_hi=hi,
                is_int=is_int,
            )
        )
    return out


def cmd_tiny(a) -> int:
    from discopt.modeling.core import from_nl

    _assert_discopt_under_test()
    hdr = _header()
    print("header", json.dumps(hdr), flush=True)
    insts = tiny_instances(a.count, a.seed)
    models = []
    with tempfile.TemporaryDirectory() as td:
        for inst in insts:
            nl = os.path.join(td, f"{inst.name}.nl")
            L.write_nl(inst, nl)
            models.append(from_nl(nl))
    import warnings

    warnings.simplefilter("ignore")  # the large-bound UserWarning is irrelevant: boxes are [0,10]

    def solve_on(backend, mdl):
        # The backend flag is re-read on every solve, so one process times both routes.
        os.environ["DISCOPT_LP_MILP_BACKEND"] = backend
        t0 = time.perf_counter()
        res = mdl.solve(time_limit=a.time_limit)
        return res, time.perf_counter() - t0

    prior = os.environ.get("DISCOPT_LP_MILP_BACKEND")
    for _ in range(2):  # warm every path
        solve_on("rust", models[1])
        solve_on("highs", models[1])
        solve_arrays(insts[1], a.time_limit)
    r_med, hh_med, h_med, compared, mismatches = [], [], [], 0, []
    orders = [("R", "H", "H0"), ("H0", "R", "H"), ("H", "H0", "R")]
    try:
        for i, (inst, mdl) in enumerate(zip(insts, models, strict=True)):
            rw, hhw, hw = [], [], []
            for rnd in range(a.rounds):
                for arm in orders[rnd % len(orders)]:
                    if arm == "R":
                        r, w = solve_on("rust", mdl)
                        rw.append(w)
                        r_obj, r_st = r.objective, r.status
                    elif arm == "H":
                        hh, w = solve_on("highs", mdl)
                        hhw.append(w)
                        hh_obj, hh_st = hh.objective, hh.status
                    else:
                        h = solve_arrays(inst, a.time_limit)
                        hw.append(h["wall"])
                        h_obj, h_st = h["objective"], h["status"]
            compared += 1
            if (
                r_st != "optimal"
                or hh_st != "optimal"
                or h_st != "Optimal"
                or abs(r_obj - h_obj) > _tol(h_obj, GAP)
                or abs(hh_obj - h_obj) > _tol(h_obj, GAP)
            ):
                mismatches.append((inst.name, r_st, r_obj, hh_st, hh_obj, h_st, h_obj))
            r_med.append(statistics.median(rw))
            hh_med.append(statistics.median(hhw))
            h_med.append(statistics.median(hw))
            if (i + 1) % 20 == 0:
                print(f"[tiny {i + 1}/{len(insts)}] R med {statistics.median(r_med) * 1e3:.2f} ms "
                      f"H med {statistics.median(hh_med) * 1e3:.2f} ms "
                      f"H0 med {statistics.median(h_med) * 1e3:.2f} ms "
                      f"load {os.getloadavg()[0]:.1f}",
                      flush=True)  # fmt: skip
    finally:
        if prior is None:
            os.environ.pop("DISCOPT_LP_MILP_BACKEND", None)
        else:
            os.environ["DISCOPT_LP_MILP_BACKEND"] = prior
    ratio = [r / h for r, h in zip(r_med, h_med, strict=True)]
    h_over = [hh - h for hh, h in zip(hh_med, h_med, strict=True)]
    res = {
        "header": hdr,
        "load_end": os.getloadavg(),
        "count": len(insts),
        "compared": compared,
        "mismatches": mismatches,
        "R_median_ms": statistics.median(r_med) * 1e3,
        "H0_median_ms": statistics.median(h_med) * 1e3,
        "R_sd_ms": statistics.stdev(r_med) * 1e3,
        "H0_sd_ms": statistics.stdev(h_med) * 1e3,
        "R_over_H0_median": statistics.median(ratio),
        "H_median_ms": statistics.median(hh_med) * 1e3,
        "H_sd_ms": statistics.stdev(hh_med) * 1e3,
        "H_minus_H0_median_ms": statistics.median(h_over) * 1e3,
        "H_minus_H0_max_ms": max(h_over) * 1e3,
        "LP": {
            "R_median_ms": statistics.median(r_med[0::2]) * 1e3,
            "H0_median_ms": statistics.median(h_med[0::2]) * 1e3,
        },
        "MILP": {
            "R_median_ms": statistics.median(r_med[1::2]) * 1e3,
            "H0_median_ms": statistics.median(h_med[1::2]) * 1e3,
        },
    }
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("tiny:", json.dumps(res, default=str), flush=True)
    return 0 if compared > 0 else 1


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("sources", nargs="+")
    s.add_argument("--panel", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-nnz", type=int, default=20000)
    s.add_argument("--milp-only", action="store_true")
    s.add_argument("--time-limit", type=float, default=60.0)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True)
    r.add_argument("--panel", action="append")
    r.add_argument("--instances", default="")
    r.add_argument("--arms", default="R,H,H0")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--time-limit", type=float, default=20.0)
    r.add_argument("--gap", type=float, default=GAP)
    r.add_argument("--max-nodes", type=int, default=100_000_000)
    r.add_argument("--out", default="")
    t = sub.add_parser("tiny")
    t.add_argument("--count", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--time-limit", type=float, default=10.0)
    t.add_argument("--out", default="")
    a = p.parse_args(argv)
    return {"screen": cmd_screen, "run": cmd_run, "tiny": cmd_tiny}[a.cmd](a)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _, _, path, arm, tl, gap, mn = sys.argv
        res = worker(path, arm, float(tl), float(gap), int(mn))
        print("RESULT " + json.dumps(res, default=str), flush=True)
    else:
        sys.exit(main())
