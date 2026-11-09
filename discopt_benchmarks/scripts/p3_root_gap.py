"""Gap plan P3 entry instrument: the root dual gap of the pure-MILP route, per instance.

Root only (``max_nodes=2``), one fresh subprocess per solve through ``lp_milp_panel.py
worker`` (the same route and marker checks the D2 panels use). The gap is
``|opt - bound| / max(1, |opt|)`` in percent against the manifest optimum, and the
bound is also checked against that optimum in the objective's own sense (bar 1: a root
bound past the optimum is a false bound and fails the run).

Usage::

    python -u discopt_benchmarks/scripts/p3_root_gap.py \
        --instances beavma,nexp-50-20-1-1 [--env DISCOPT_X=1] [--out root.json]

Prints one line per instance and an executed-comparison count; exits non-zero when no
comparison ran or a bound crossed its optimum (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
PANEL_PY = HERE.parent / "lp_milp_panel.py"
sys.path.insert(0, str(HERE.parent))

import p42_rens_panel as R  # noqa: E402, N812  (population: path, opt, sense)


def root_solve(path: str, extra_env: dict[str, str], time_limit: float) -> dict:
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", DISCOPT_LP_MILP_BACKEND="rust")
    env.update(extra_env)
    cmd = [sys.executable, str(PANEL_PY), "worker", path, "R", str(time_limit), "1e-4", "2"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=time_limit + 300)
    if p.returncode != 0:
        raise RuntimeError(f"{path}: rc={p.returncode} {p.stderr[-1500:]}")
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        raise RuntimeError(f"{path}: no RESULT line\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    return json.loads(lines[-1][len("RESULT ") :])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the solve")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    extra = dict(kv.split("=", 1) for kv in a.env)
    want = a.instances.split(",")
    pop = {m["name"]: m for m in R.population()}
    missing = [w for w in want if w not in pop]
    if missing:
        raise SystemExit(f"not in the panel population: {missing}")
    rows, compared, crossed = [], 0, []
    for name in want:
        mem = pop[name]
        r = root_solve(str(REPO / mem["path"]), extra, a.time_limit)
        opt, bound, maximize = mem.get("oracle_objective"), r.get("bound"), mem["maximize"]
        gap = None
        if opt is not None and bound is not None:
            compared += 1
            past = bound < opt - 1e-6 * max(1.0, abs(opt)) if maximize else (
                bound > opt + 1e-6 * max(1.0, abs(opt))
            )
            if past:
                crossed.append(name)
            gap = 100.0 * abs(opt - bound) / max(1.0, abs(opt))
        row = {
            "name": name,
            "opt": opt,
            "bound": bound,
            "root_gap_pct": gap,
            "nodes": r.get("nodes"),
            "wall": r.get("wall"),
            "status": r.get("status"),
            "env": extra,
        }
        rows.append(row)
        g = "n/a" if gap is None else f"{gap:.3f}%"
        print(f"{name:24} bound={bound} opt={opt} root_gap={g} wall={r.get('wall'):.1f}s",
              flush=True)  # fmt: skip
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    print(f"executed comparisons: {compared}; bounds past the optimum: {crossed}", flush=True)
    return 0 if compared and not crossed else 1


if __name__ == "__main__":
    sys.exit(main())
