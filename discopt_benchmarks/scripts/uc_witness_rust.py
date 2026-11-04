"""highs-parity P0.7: the #1667 unit-commitment witness on the Rust MILP driver (M-1671 entry).

``docs/dev/highs-parity-plan.md`` P0.7. The model is ``UC_SCRIPT`` from
``python/tests/test_1667_highs_sparsify_presolve_cycle.py`` (the HiGHS-presolve cycle
witness, 2904 x 4344, 240 binaries), with its ``m.solve(time_limit=20)`` replaced per arm.
Each solve is a fresh subprocess, interleaved per repetition:

* ``rust@T`` for T in --limits: ``DISCOPT_LP_MILP_BACKEND=rust``;
* ``rust-nopresolve@A``: the same with ``presolve=False``;
* ``rust-nocmir@A``: the same with ``DISCOPT_MILP_CMIR=0``;
* ``highs@20``: the shipped default route, the reference (the test's own arm).

A is --attr-limit. Each row records status, objective, bound, gap_certified, nodes, wall and
the 1-min load at start and end. Kill (P0.7): a Rust bound above the known optimum
``450623.7692`` (plus 1e-6 relative) or an ``optimal`` objective off it -> a correctness
defect; the script counts both and exits non-zero on any.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
UC_OPT = 450623.7692
_SOLVE = "res = m.solve(time_limit=20)"


def uc_script(time_limit: float, presolve: bool) -> str:
    spec = importlib.util.spec_from_file_location(
        "t1667", REPO / "python" / "tests" / "test_1667_highs_sparsify_presolve_cycle.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = mod.UC_SCRIPT
    if src.count(_SOLVE) != 1:
        raise RuntimeError("UC_SCRIPT no longer has exactly one solve line to replace")
    src = src.replace(_SOLVE, f"res = m.solve(time_limit={time_limit!r}, presolve={presolve!r})")
    src = src.replace(
        'print("RESULT", res.status, repr(res.objective), repr(res.bound), wall, flush=True)',
        'import json as _j; print("RESULT " + _j.dumps({"status": str(res.status), '
        '"objective": res.objective, "bound": res.bound, "certified": bool(res.gap_certified), '
        '"nodes": int(res.node_count), "wall": wall, "route": res.algorithm_route}), flush=True)',
    )
    if "RESULT " not in src:
        raise RuntimeError("UC_SCRIPT's RESULT line changed; update this script")
    return src


def run_arm(arm: str, time_limit: float, presolve: bool, env_extra: dict) -> dict:
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    load0 = os.getloadavg()[0]
    p = subprocess.run([sys.executable, "-u", "-c", uc_script(time_limit, presolve)], env=env,
                       capture_output=True, text=True, timeout=10 * time_limit + 300)  # fmt: skip
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    files = [ln for ln in p.stdout.splitlines() if ln.startswith("FILE ")]
    if p.returncode != 0 or len(lines) != 1:
        return {"arm": arm, "error": f"rc={p.returncode}", "stderr": p.stderr[-1500:]}
    if not files or not files[0].split()[1].startswith(str(REPO / "python")):
        raise RuntimeError(f"solved with a discopt outside this tree: {files}")
    r = json.loads(lines[0][len("RESULT ") :])
    return {"arm": arm, "time_limit": time_limit, **r, "load": [load0, os.getloadavg()[0]]}


def violations(r: dict) -> list[str]:
    tol = 1e-6 * abs(UC_OPT)
    why = []
    if r.get("bound") is not None and r["bound"] > UC_OPT + tol:
        why.append(f"bound {r['bound']} above the optimum {UC_OPT}")
    if "optimal" in str(r.get("status", "")).lower() and (
        r.get("objective") is None or abs(r["objective"] - UC_OPT) > 1e-4 * abs(UC_OPT) + tol
    ):
        why.append(f"optimal objective {r.get('objective')} vs {UC_OPT}")
    return why


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--limits", default="20,60,120")
    p.add_argument("--attr-limit", type=float, default=60.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    arms = [(f"rust@{t:g}", float(t), True, {"DISCOPT_LP_MILP_BACKEND": "rust"})
            for t in map(float, a.limits.split(","))]  # fmt: skip
    arms += [
        (f"rust-nopresolve@{a.attr_limit:g}", a.attr_limit, False,
         {"DISCOPT_LP_MILP_BACKEND": "rust"}),
        (f"rust-nocmir@{a.attr_limit:g}", a.attr_limit, True,
         {"DISCOPT_LP_MILP_BACKEND": "rust", "DISCOPT_MILP_CMIR": "0"}),
        ("highs@20", 20.0, True, {}),
    ]  # fmt: skip
    rows, executed, bad = [], 0, []
    print(f"header load={os.getloadavg()} arms={[x[0] for x in arms]} reps={a.reps}", flush=True)
    for rep in range(a.reps):
        order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
        for arm, tl, pre, env in order:
            t = time.perf_counter()
            r = run_arm(arm, tl, pre, env)
            r["rep"] = rep
            rows.append(r)
            if "error" not in r:
                executed += 1
                why = violations(r)
                r["violations"] = why
                bad += [f"{arm} rep {rep}: {w}" for w in why]
            print(f"[rep {rep}] {arm:22s} {r.get('status')} obj={r.get('objective')} "
                  f"bound={r.get('bound')} cert={r.get('certified')} nodes={r.get('nodes')} "
                  f"wall={r.get('wall', float('nan')):.1f} ({time.perf_counter() - t:.0f}s) "
                  f"{r.get('error', '')}", flush=True)  # fmt: skip
            Path(a.out).write_text(json.dumps({"rows": rows, "executed": executed,
                                               "violations": bad}, indent=1) + "\n")  # fmt: skip
    print(f"uc: executed={executed} violations={len(bad)}", flush=True)
    for b in bad:
        print("  !!!", b, flush=True)
    return 0 if executed and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
