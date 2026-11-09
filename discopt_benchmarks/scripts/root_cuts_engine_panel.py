"""highs-parity P1.4: NLP-BB root cuts on the Rust simplex vs HiGHS (flag A/B).

Arms: ``DISCOPT_NLPBB_ROOT_CUTS_ENGINE=highs`` (OFF, the default) and ``=rust`` (ON). The
population is every in-repo corpus instance on which the root-cut stage fires, in two
configurations: the default, and ``DISCOPT_CONVEX_KERNEL=0 DISCOPT_CONVEX_MINLP_ROUTE=0``
(``@nlpbb``: the kernel and the convex route otherwise answer most convex models first).
It is found by ``screen``, which counts ``generate_root_cuts`` calls; the flag changes
nothing anywhere else by construction.

``run`` solves each (instance, configuration) in each arm, in a fresh subprocess, arms
interleaved with rotated order, ``--reps`` times. Checks per row: the bound against the
``minlplib.solu`` optimum (fetched into ``ref/minlplib/``) and against every arm's
verified incumbent; every incumbent with ``verify_point``; a certified point that fails
verification. Prints executed rows and checks; exits non-zero on 0 executed or any
violation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
DATA = REPO / "python" / "tests" / "data"
NLPBB_ENV = {"DISCOPT_CONVEX_KERNEL": "0", "DISCOPT_CONVEX_MINLP_ROUTE": "0"}
CONFIGS = {"default": {}, "nlpbb": NLPBB_ENV}

_CHILD = r"""
import sys, json, time
import numpy as np
import discopt.solvers._root_cuts as rc
n = {"stage": 0}
real = rc.generate_root_cuts
def w(*a, **k):
    n["stage"] += 1
    return real(*a, **k)
rc.generate_root_cuts = w
from discopt.modeling.core import from_nl
from discopt.validation.feasibility import verify_point
m = from_nl(sys.argv[1])
t = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2]))
wall = time.perf_counter() - t
ok = None
if r.x is not None:
    flat = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
    ok = bool(verify_point(m, flat).ok)
from discopt.modeling.core import ObjectiveSense
print("RESULT " + json.dumps({"status": str(r.status), "cert": bool(r.gap_certified),
      "obj": r.objective, "bound": r.bound, "nodes": int(r.node_count), "wall": wall,
      "feas_ok": ok, "stage_calls": n["stage"],
      "sense": 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0}), flush=True)
"""


def _corpus() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in (DATA / "minlplib_nl", DATA / "minlplib"):
        for p in sorted(d.glob("*.nl")):
            out.setdefault(p.stem, p)
    return out


def _solu() -> dict[str, float]:
    sys.path.insert(0, str(HERE.parent))
    import master_parity_panel as mp

    return mp._solu()


def _child(path: Path, tl: float, env_extra: dict) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    try:
        p = subprocess.run([sys.executable, "-c", _CHILD, str(path), str(tl)], env=env,
                           capture_output=True, text=True, timeout=4 * tl + 120)  # fmt: skip
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"error": f"rc={p.returncode}: {(p.stderr.strip().splitlines() or [''])[-1][:300]}"}
    return json.loads(lines[0][len("RESULT ") :])


def cmd_screen(a) -> int:
    paths = _corpus()
    found, ran = [], 0
    for cfg, env in CONFIGS.items():
        for name, p in paths.items():
            r = _child(p, a.time_limit, {**env, "DISCOPT_NLPBB_ROOT_CUTS_ENGINE": "highs"})
            ran += "error" not in r
            if r.get("stage_calls", 0) > 0:
                found.append({"instance": name, "config": cfg})
            print(f"[screen {cfg}] {name:24s} stage_calls={r.get('stage_calls')} "
                  f"{r.get('error', '')}", flush=True)  # fmt: skip
    Path(a.out).write_text(json.dumps({"population": found, "screened": ran}, indent=1) + "\n")
    print(f"screen: screened={ran} population={len(found)}", flush=True)
    return 0 if ran else 1


def cmd_run(a) -> int:
    pop = json.loads(Path(a.population).read_text())["population"]
    paths, solu = _corpus(), _solu()
    arms = ["highs", "rust"]
    rows, checks, viol, errors = [], 0, [], []
    for i, item in enumerate(pop, 1):
        name, cfg = item["instance"], item["config"]
        per: dict[str, list] = {arm: [] for arm in arms}
        for rep in range(a.reps):
            for arm in arms if rep % 2 == 0 else arms[::-1]:
                env = {**CONFIGS[cfg], "DISCOPT_NLPBB_ROOT_CUTS_ENGINE": arm}
                r = _child(paths[name], a.time_limit, env)
                if "error" in r:
                    errors.append(f"{name}@{cfg} [{arm}] {r['error']}")
                per[arm].append(r)
        refs = [("solu", solu[name])] if name in solu else []
        for arm in arms:
            for r in per[arm]:
                if r.get("feas_ok") and r.get("obj") is not None:
                    refs.append((f"inc[{arm}]", float(r["obj"])))
        for arm in arms:
            for r in per[arm]:
                if "error" in r:
                    continue
                s = r["sense"]
                if r.get("bound") is not None:
                    for lab, ref in refs:
                        checks += 1
                        if s * r["bound"] > s * ref + 1e-6 * max(1.0, abs(ref)):
                            viol.append(f"{name}@{cfg} [{arm}] bound {r['bound']} past {lab} {ref}")
                if r.get("cert") and r.get("feas_ok") is False:
                    viol.append(f"{name}@{cfg} [{arm}] certified an unverified point")
        summ = {"instance": name, "config": cfg}
        for arm in arms:
            ok = [r for r in per[arm] if "error" not in r]
            summ[arm] = {
                "cert": [r["cert"] for r in ok], "bound": [r["bound"] for r in ok],
                "obj": [r["obj"] for r in ok], "nodes": [r["nodes"] for r in ok],
                "wall_med": statistics.median([r["wall"] for r in ok]) if ok else None,
            }  # fmt: skip
        rows.append(summ)
        print(f"[run {i}/{len(pop)}] {name}@{cfg}: " + "  ".join(
            f"{arm} cert={summ[arm]['cert']} nodes={summ[arm]['nodes']} "
            f"bound={summ[arm]['bound'][:1]} wall={summ[arm]['wall_med']}" for arm in arms),
            flush=True)  # fmt: skip
        doc = {"rows": rows, "checks": checks, "violations": viol, "errors": errors}
        Path(a.out).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    executed = sum(len(r[arm]["cert"]) for r in rows for arm in arms)
    print(f"run: rows={len(rows)} executed={executed} checks={checks} "
          f"violations={len(viol)} errors={len(errors)}", flush=True)  # fmt: skip
    for v in viol + errors:
        print("FLAG", v, flush=True)
    return 0 if executed and not viol else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("--out", required=True)
    s.add_argument("--time-limit", type=float, default=20.0)
    r = sub.add_parser("run")
    r.add_argument("--population", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--time-limit", type=float, default=60.0)
    a = p.parse_args(argv)
    t = time.perf_counter()
    rc = {"screen": cmd_screen, "run": cmd_run}[a.cmd](a)
    print(f"elapsed {time.perf_counter() - t:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
