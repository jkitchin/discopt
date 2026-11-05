"""LP/MILP routing panel: R (Rust route) vs H (HiGHS route) vs H0 (raw highspy).

Stage 0 of ``docs/dev/lp-milp-highs-routing-plan.md`` (§6). Subcommands:

``fetch``   clone ERGO-Code/HiGHS at the tag of the installed ``highspy`` into
            ``ref/HiGHS`` (git-ignored) and record the tag and commit in the manifest's
            ``source`` block. The L/M/INF panels are screened from its
            ``check/instances``.
``screen``  build a panel manifest: read each instance, validate the loader (raw HiGHS on
            the original file = oracle; raw HiGHS on the converted arrays must reproduce
            it; the ``.nl`` round-trip must load with the right shape). Mismatches are
            *excluded and listed*, never dropped silently. ``--kind`` selects the class:
            ``lp`` (no integers, oracle optimal; panel L), ``milp`` (integers, oracle
            optimal; panel M), ``inf`` (oracle infeasible or unbounded; panel INF).
``run``     solve manifest instances with the requested arms, rounds interleaved per
            instance with the arm order rotated each round. R/H run ``Model.solve`` in a
            subprocess with ``DISCOPT_LP_MILP_BACKEND=rust|highs``; H0 runs raw highspy
            in-process. Every R/H row must carry the route marker of its arm (a row
            answered by the other engine raises: it would be a measurement of the wrong
            route). ``--load-gate G`` marks the walls of any instance whose 1-min load
            exceeded ``G`` at its start or end as not timing-valid (``wall_valid``
            false on the row, ``*`` in the progress line); they are still recorded so
            a report can show them as indicative. Statuses and nodes do not depend on
            load.
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
HIGHS_REPO = "https://github.com/ERGO-Code/HiGHS"
REF_DIR = REPO / "ref" / "HiGHS"
#: The ``DISCOPT_LP_MILP_BACKEND`` value and route marker each subprocess arm must show.
ARM_BACKEND = {"R": "rust", "H": "highs", "S": "highs"}
#: The route marker each subprocess arm must show. ``S`` (highs-parity P2.6) is the verified
#: LP route with the Rust engine behind P1.1's seam (``DISCOPT_LP_STD_ENGINE=rust``): the
#: same dispatch and certificate ladder as ``H``, the raw LP solved by the Rust simplex.
#: Pure LP only -- a MILP still goes to HiGHS on that route, and ``check_route`` refuses it.
ARM_ROUTE = {"R": "rust", "H": "highs", "S": "seam-rust"}
#: ``DISCOPT_LP_STD_ENGINE`` per arm (``None``: unset, the HiGHS default).
ARM_STD_ENGINE = {"R": None, "H": None, "S": "rust"}
#: HiGHS oracle statuses that put an instance on the INF panel.
INF_STATUSES = ("Infeasible", "Unbounded", "Primal infeasible or unbounded")


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


# ----------------------------------------------------------------------------- fetch
def _highspy_version() -> str:
    import highspy

    v = getattr(highspy, "__version__", None)
    if not v:
        from importlib.metadata import version

        v = version("highspy")
    return str(v)


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def cmd_fetch(a) -> int:
    tag = a.tag or f"v{_highspy_version()}"
    dest = Path(a.dest)
    if dest.exists():
        tags = (
            _git("tag", "--points-at", "HEAD", cwd=dest).split() if (dest / ".git").exists() else []
        )
        if tag not in tags:
            raise SystemExit(f"{dest} exists with tags {tags}, wanted {tag}; remove it or "
                             "pass --tag")  # fmt: skip
        print(f"fetch: {dest} already at {tag}", flush=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch: cloning {HIGHS_REPO} @ {tag} into {dest}", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", tag, HIGHS_REPO, str(dest)], check=True
        )
    commit = _git("rev-parse", "HEAD", cwd=dest)
    inst = dest / "check" / "instances"
    n_files = sum(1 for q in inst.iterdir() if q.suffix in (".mps", ".lp"))
    if n_files == 0:
        raise SystemExit(f"{inst} holds no .mps/.lp files")
    out = Path(a.manifest)
    man = json.loads(out.read_text()) if out.exists() else {"members": [], "excluded": []}
    man["source"] = {
        "repo": HIGHS_REPO,
        "tag": tag,
        "commit": commit,
        "path": os.path.relpath(inst, REPO),
        "files": n_files,
    }
    out.write_text(json.dumps(man, indent=1) + "\n")
    print(f"fetch: tag={tag} commit={commit} files={n_files}", flush=True)
    return 0


# ----------------------------------------------------------------------------- screen
def _kind_reject(kind: str, n_int: int) -> str | None:
    """Reason an instance is outside the ``--kind`` class before its oracle runs."""
    if kind == "lp" and n_int > 0:
        return "has integer columns"
    if kind == "milp" and n_int == 0:
        return "no integer columns"
    return None


def cmd_screen(a) -> int:
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
            why = _kind_reject(a.kind, n_int)
            if why:
                excluded.append({"file": str(f), "reason": why})
                print(tag, "EXCLUDE", why, flush=True)
                continue
            if nnz > a.max_nnz:
                excluded.append({"file": str(f), "reason": f"nnz {nnz} > {a.max_nnz}"})
                print(tag, f"EXCLUDE nnz={nnz}", flush=True)
                continue
            ref = L.highs_reference(str(f), time_limit=a.time_limit)
            if a.kind == "inf":
                if ref["status"] not in INF_STATUSES:
                    excluded.append({"file": str(f), "reason": f"oracle status {ref['status']}"})
                    print(tag, "EXCLUDE oracle", ref["status"], flush=True)
                    continue
                h0 = solve_arrays(inst, a.time_limit)
                compared += 1
                if h0["status"] != ref["status"]:
                    excluded.append({"file": str(f),
                                     "reason": f"array mismatch: {h0['status']} vs oracle "
                                               f"{ref['status']}"})  # fmt: skip
                    print(tag, "EXCLUDE array mismatch", h0["status"], ref["status"], flush=True)
                    continue
                members.append(_roundtrip_member(inst, f, a.panel, td, nnz, n_int, ref, None))
                print(tag, f"OK n={inst.n} m={inst.m} nnz={nnz} int={n_int} {ref['status']}",
                      flush=True)  # fmt: skip
                continue
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
            try:
                row = _roundtrip_member(inst, f, a.panel, td, nnz, n_int, ref, h0["x"])
            except ValueError as e:
                excluded.append({"file": str(f), "reason": f"nl writer: {e}"})
                print(tag, "EXCLUDE nl writer:", e, flush=True)
                continue
            members.append(row)
            print(tag, f"OK n={inst.n} m={inst.m} nnz={nnz} int={n_int} obj={ref['objective']}",
                  flush=True)  # fmt: skip
    out = Path(a.out)
    old = json.loads(out.read_text()) if out.exists() else {"members": [], "excluded": []}
    keep = [x for x in old["members"] if x["panel"] != a.panel]
    keep_ex = [x for x in old["excluded"] if x.get("panel") != a.panel]
    for x in excluded:
        x["panel"] = a.panel
    old["members"] = keep + members
    old["excluded"] = keep_ex + excluded
    out.write_text(json.dumps(old, indent=1) + "\n")
    for x in excluded:
        print(f"  excluded [{a.panel}] {os.path.basename(x['file'])}: {x['reason']}", flush=True)
    print(f"screen: panel={a.panel} members={len(members)} excluded={len(excluded)} "
          f"compared={compared}", flush=True)  # fmt: skip
    return 0 if compared > 0 else 1


def _roundtrip_member(inst, f, panel, td, nnz, n_int, ref, x_h0) -> dict:
    """Write the ``.nl``, load it back, check its shape; return the manifest row.

    ``ValueError`` from the ``.nl`` writer (an explicit refusal) propagates to the caller,
    which excludes the instance with the reason; a shape mismatch is a loader bug and
    raises ``RuntimeError``.
    """
    from discopt.modeling.core import from_nl

    nl = os.path.join(td, f"{inst.name}.nl")
    L.write_nl(inst, nl)
    mdl = from_nl(nl)
    nv = sum(v.size for v in mdl._variables)
    if nv != inst.n:
        raise RuntimeError(f"{inst.name}: .nl round-trip has {nv} vars, expected {inst.n}")
    feasible = ref["status"] == "Optimal"
    return {
        "name": inst.name,
        "path": os.path.relpath(f, REPO),
        "panel": panel,
        "n": inst.n,
        "m": inst.m,
        "nnz": nnz,
        "n_int": n_int,
        "oracle_status": ref["status"],
        "oracle_objective": ref["objective"] if feasible else None,
        "oracle_nodes": ref["nodes"],
        "oracle_max_viol": L.dense_max_violation(inst, x_h0) if feasible else None,
    }


# ----------------------------------------------------------------------------- run
def solve_h0(inst: L.LinearInstance, time_limit: float, gap: float) -> dict:
    """H0 row: raw highspy on the arrays, verified by the script's own dense check."""
    r = solve_arrays(inst, time_limit, gap)
    x_h0 = r.pop("x")
    r["max_viol"] = None if x_h0 is None else L.dense_max_violation(inst, x_h0)
    r["status"] = {"Optimal": "optimal"}.get(r["status"], r["status"])
    return r


def worker(path: str, arm: str, time_limit: float, gap: float, max_nodes: int) -> dict:
    if arm == "H0":
        return solve_h0(L.read_instance(path), time_limit, gap)
    import discopt.solver as solver_mod
    from discopt.modeling.core import from_nl
    from discopt.status import TERMINATION_PRESOLVE

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
        # The HiGHS route sets this marker at its dispatch point (``lp_milp_highs``); any
        # other engine leaves it unset. A HiGHS-arm solve the route declined (and passed
        # on to the Rust engine) therefore reports "rust", and ``check_route`` refuses it.
        # A solve root presolve answered before the LP/MILP dispatch (an infeasibility
        # proof, #1618 C-17) ran neither engine and says so: "presolve".
        "route": "presolve"
        if r.termination == TERMINATION_PRESOLVE
        else "highs"
        if stats.get("route/lp_milp_backend") == 1.0
        else "seam-rust"
        if stats.get("lp/std_engine_rust") == 1.0
        else "rust",
        "algorithm_route": r.algorithm_route,
        "max_viol": viol,
        "env_backend": os.environ.get("DISCOPT_LP_MILP_BACKEND"),
        "env_std_engine": os.environ.get("DISCOPT_LP_STD_ENGINE"),
    }


def check_route(name: str, arm: str, res: dict) -> None:
    """Refuse a subprocess-arm row that was answered by the other engine's route.

    A row whose route marker does not match its arm would be reported as a measurement
    of the wrong engine (CLAUDE.md §8), so it raises instead of being recorded. A row
    root presolve answered before the dispatch (route ``"presolve"``) ran *neither*
    engine; it is kept -- its verdict is still checked against the oracle -- and
    reported as such, never as a measurement of the arm's engine.
    """
    want, env = ARM_ROUTE[arm], ARM_BACKEND[arm]
    if (
        res.get("env_backend") != env
        or res.get("env_std_engine") != ARM_STD_ENGINE[arm]
        or res.get("route") not in (want, "presolve")
    ):
        raise RuntimeError(
            f"{name}: arm {arm} expects route {want!r} (env {env!r}, std engine "
            f"{ARM_STD_ENGINE[arm]!r}), got route {res.get('route')!r} with env "
            f"{res.get('env_backend')!r} / {res.get('env_std_engine')!r} "
            f"(algorithm_route {res.get('algorithm_route')!r})"
        )


def arm_order(arms: list[str], rnd: int) -> list[str]:
    """Round ``rnd``'s arm order: ``arms`` rotated left by ``rnd`` (each arm leads once)."""
    k = rnd % len(arms)
    return arms[k:] + arms[:k]


def load_ok(gate: float | None, *loads: float) -> bool:
    """Whether walls may be reported: every sampled 1-min load is at or under the gate."""
    return gate is None or all(x <= gate for x in loads)


def _run_sub(path: str, arm: str, a) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if arm in ARM_BACKEND:
        env["DISCOPT_LP_MILP_BACKEND"] = ARM_BACKEND[arm]
        env.pop("DISCOPT_LP_STD_ENGINE", None)
        if ARM_STD_ENGINE[arm] is not None:
            env["DISCOPT_LP_STD_ENGINE"] = ARM_STD_ENGINE[arm]
    cmd = [sys.executable, str(HERE), "worker", path, arm, str(a.time_limit), str(a.gap),
           str(a.max_nodes)]  # fmt: skip
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3 * a.time_limit + 120)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"error": f"rc={p.returncode}", "stderr_tail": p.stderr[-1500:]}
    return json.loads(lines[0][len("RESULT ") :])


def _incorrect_inf(res: dict, oracle_status: str) -> list[str]:
    """INF panel: a returned point or a contradicting verdict is wrong; a refusal is not."""
    st = str(res.get("status", "")).lower()
    why = []
    if st in ("optimal", "feasible"):
        why.append(f"status {res.get('status')} on an instance HiGHS calls {oracle_status}")
    if oracle_status == "Infeasible" and st == "unbounded":
        why.append("unbounded on an infeasible instance")
    if oracle_status == "Unbounded" and st == "infeasible":
        why.append("infeasible on an unbounded instance")
    return why


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
    bad = [arm for arm in arms if arm not in (*ARM_BACKEND, "H0")]
    if bad:
        raise SystemExit(f"unknown arms {bad}; expected a subset of R,S,H,H0")
    hdr = _header()
    hdr["load_gate"] = a.load_gate
    print("header", json.dumps(hdr), "arms", arms, "n", len(mem), flush=True)
    rows: list[dict] = []
    if a.resume:
        rows = resume_rows(Path(a.out), hdr["commit"], vars(a))
        # Keep every earlier segment's header (commit, start load): the run's identity is
        # the first segment's commit, and each segment's load belongs in the record.
        old_hdr = json.loads(Path(a.out).read_text())["header"]
        hdr["resumed_from"] = [*old_hdr.pop("resumed_from", []), old_hdr]
        print(f"resume: {len(rows)} instance rows already in {a.out}", flush=True)
    done = {r["name"] for r in rows}
    for i, x in enumerate(mem, 1):
        if x["name"] in done:
            continue
        path = str(REPO / x["path"])
        inst = L.read_instance(path)
        oracle_status = x.get("oracle_status", "Optimal")
        per: dict[str, list] = {arm: [] for arm in arms}
        n_exec = n_wrong = 0
        load_start = os.getloadavg()[0]
        for rnd in range(a.rounds):
            for arm in arm_order(arms, rnd):
                if arm == "S" and int(inst.is_int.sum()) > 0:
                    # The seam arm is the pure-LP route only; a MILP on it goes to HiGHS
                    # and would be a measurement of the wrong engine (ARM_ROUTE).
                    r = {"skipped": "arm S is the pure-LP route; this instance has integers"}
                elif arm == "H0" and not a.h0_subprocess:
                    r = solve_h0(inst, a.time_limit, a.gap)
                elif arm == "H0":
                    # A fresh process per solve: HiGHS's scheduler and any state a previous
                    # solve left behind are not shared across rows (P0.2).
                    r = _run_sub(path, arm, a)
                else:
                    r = _run_sub(path, arm, a)
                    if "skipped" not in r and "error" not in r:
                        check_route(x["name"], arm, r)
                if "skipped" in r or "error" in r:
                    per[arm].append(r)
                    continue
                if oracle_status == "Optimal":
                    r["incorrect"] = _incorrect(r, x["oracle_objective"], inst.maximize, a.gap)
                else:
                    r["incorrect"] = _incorrect_inf(r, oracle_status)
                n_exec += 1
                n_wrong += bool(r["incorrect"])
                per[arm].append(r)
        load_end = os.getloadavg()[0]
        wall_ok = load_ok(a.load_gate, load_start, load_end)
        summ = {"name": x["name"], "panel": x["panel"], "oracle": x["oracle_objective"],
                "oracle_status": oracle_status, "load": [load_start, load_end],
                "wall_valid": wall_ok, "executed": n_exec,
                "incorrect_solves": n_wrong}  # fmt: skip
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
                # Walls are always recorded; ``wall_valid`` on the row says whether the
                # load gate held, and a report must label a gated wall indicative.
                "wall_med": statistics.median(walls),
                "wall_sd": statistics.stdev(walls) if len(walls) > 1 else 0.0,
                "walls_all": walls,
                "status_all": [r.get("status") for r in ok],
                "certified_all": [r.get("gap_certified") for r in ok],
                "nodes_all": [r.get("nodes") for r in ok],
                "routes": sorted({str(r.get("route")) for r in ok if "route" in r}),
                "incorrect": sorted({w for r in ok for w in r["incorrect"]}),
            }  # fmt: skip
        rows.append(summ)
        brief = {arm: (v.get("status"), v.get("nodes"), v.get("objective"),
                       f"{v['wall_med']:.3f}{'' if wall_ok else '*'}"
                       if "wall_med" in v else v)
                 for arm, v in summ.items() if arm in arms}  # fmt: skip
        gated = "" if wall_ok else "  (* wall indicative: load over the gate)"
        print(f"[run {i}/{len(mem)}] {x['panel']:4s} {x['name']:24s} {brief}{gated}", flush=True)
        for arm in arms:
            if "presolve" in summ[arm].get("routes", ()):
                print(f"  note {arm}: answered by root presolve, neither engine ran", flush=True)
            if summ[arm].get("incorrect"):
                print(f"  !!! INCORRECT {arm}: {summ[arm]['incorrect']}", flush=True)
        if a.out:  # after every instance, so a killed run loses at most one (CLAUDE.md §10)
            _write_run(Path(a.out), hdr, vars(a), rows, arms, complete=False)
    hdr["load_end"] = os.getloadavg()
    t = tally(rows, arms)
    if a.out:
        _write_run(Path(a.out), hdr, vars(a), rows, arms, complete=True)
    for panel, n in sorted(t["executed_by_panel"].items()):
        print(f"run: panel={panel} executed={n}", flush=True)
    print(f"run: presolve_answered={t['presolve_answered']}", flush=True)
    print(f"run: executed={t['executed']} incorrect_count={t['incorrect_count']} "
          f"walls_gated={t['walls_gated']}/{len(rows)} load_end={hdr['load_end']}",
          flush=True)  # fmt: skip
    return 0 if t["executed"] > 0 else 1


#: ``run`` arguments that must match for ``--resume`` to append to an earlier run.
RESUME_KEYS = ("manifest", "panel", "instances", "arms", "rounds", "time_limit", "gap",
               "max_nodes", "h0_subprocess", "load_gate")  # fmt: skip


def tally(rows: list[dict], arms: list[str]) -> dict:
    """Run totals, recomputed from the per-instance rows (so a resumed run counts the same)."""
    by_panel: dict[str, int] = {}
    presolve: dict[str, int] = {}
    for r in rows:
        by_panel[r["panel"]] = by_panel.get(r["panel"], 0) + r["executed"]
        for arm in arms:
            if "presolve" in r.get(arm, {}).get("routes", ()):
                presolve[arm] = presolve.get(arm, 0) + 1
    return {
        "executed": sum(r["executed"] for r in rows),
        "incorrect_count": sum(r["incorrect_solves"] for r in rows),
        "executed_by_panel": by_panel,
        "walls_gated": sum(not r["wall_valid"] for r in rows),
        "presolve_answered": presolve,
    }


def _write_run(path: Path, hdr: dict, args: dict, rows: list[dict], arms, complete: bool):
    out = {"header": hdr, "args": args, "complete": complete, "rows": rows, **tally(rows, arms)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=str) + "\n")
    tmp.replace(path)


#: What a resumed run must share with the run it appends to: the solver, the instance
#: loader and the manifests. Harness-only commits (reporting, orchestration) may land in
#: between; each run JSON records its own harness commit in its header.
CODE_UNDER_TEST = ("python/discopt", "crates", "discopt_benchmarks/scripts/lp_milp_loader.py",
                   "discopt_benchmarks/data")  # fmt: skip


def code_fingerprint(commit: str) -> dict[str, str]:
    """Git tree/blob hash of each ``CODE_UNDER_TEST`` path at ``commit``."""
    return {p: _git("rev-parse", f"{commit}:{p}", cwd=REPO) for p in CODE_UNDER_TEST}


def resume_rows(path: Path, commit: str, args: dict) -> list[dict]:
    """The rows of an interrupted run, refusing one that measured different code or settings."""
    if not path.exists():
        raise SystemExit(f"--resume: {path} does not exist")
    old = json.loads(path.read_text())
    if old.get("complete"):
        raise SystemExit(f"--resume: {path} is a complete run; nothing to resume")
    old_commit = old["header"]["commit"]
    if old_commit != commit:
        dirty = _git("status", "--porcelain", "--", *CODE_UNDER_TEST, cwd=REPO)
        if dirty:
            raise SystemExit(f"--resume: uncommitted changes in the code under test:\n{dirty}")
        a, b = code_fingerprint(old_commit), code_fingerprint(commit)
        changed = sorted(k for k in a if a[k] != b[k])
        if changed:
            raise SystemExit(f"--resume: {path} was measured at {old_commit}; HEAD {commit} "
                             f"changes the code under test {changed}; a resumed run must "
                             "measure one solver")  # fmt: skip
    diff = {k: (old["args"].get(k), args.get(k)) for k in RESUME_KEYS
            if old["args"].get(k) != args.get(k)}  # fmt: skip
    if diff:
        raise SystemExit(f"--resume: run arguments differ from {path}: {diff}")
    return old["rows"]


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
    f = sub.add_parser("fetch")
    f.add_argument("--manifest", required=True)
    f.add_argument("--dest", default=str(REF_DIR))
    f.add_argument("--tag", default="", help="HiGHS git tag (default: v<installed highspy>)")
    s = sub.add_parser("screen")
    s.add_argument("sources", nargs="+")
    s.add_argument("--panel", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-nnz", type=int, default=20000)
    s.add_argument("--kind", choices=("any", "lp", "milp", "inf"), default="any")
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
    r.add_argument(
        "--resume",
        action="store_true",
        help="append to an interrupted --out run (same commit and arguments)",
    )
    r.add_argument(
        "--h0-subprocess",
        action="store_true",
        help="run the H0 arm in a fresh subprocess per solve, like R and H",
    )
    r.add_argument("--load-gate", type=float, default=None,
                   help="mark walls not timing-valid when the 1-min load exceeds this")  # fmt: skip
    t = sub.add_parser("tiny")
    t.add_argument("--count", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--time-limit", type=float, default=10.0)
    t.add_argument("--out", default="")
    a = p.parse_args(argv)
    cmds = {"fetch": cmd_fetch, "screen": cmd_screen, "run": cmd_run, "tiny": cmd_tiny}
    return cmds[a.cmd](a)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        _, _, path, arm, tl, gap, mn = sys.argv
        res = worker(path, arm, float(tl), float(gap), int(mn))
        print("RESULT " + json.dumps(res, default=str), flush=True)
    else:
        sys.exit(main())
