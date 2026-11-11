"""Pseudocost sentinel-child audit: how often does a non-bound reach ``Pseudocosts::update``?

``TreeManager::process_evaluated`` feeds every evaluated child's ``local_lower_bound``
into ``Pseudocosts::update`` before pruning. A child the Rust MILP driver found
LP-infeasible carries the ``1e30`` exclusion sentinel there, and the update used to
take ``(1e30 - parent_lb) / frac`` as a bound gain. This panel counts, per solve, the
profile counters

* ``PseudocostUpdates``          -- observations offered to ``Pseudocosts::update``;
* ``PseudocostSentinelSkipped``  -- of those, a parent or child bound that is non-finite
  or ``|b| >= 1e20`` (the LP layer's INF);
* ``PseudocostUntrustedChild``   -- observations from a child whose own relaxation
  produced no bound (reads as a zero gain; counted, not filtered),

and checks every result against the reference optimum:

  (a) a ``gap_certified`` objective equals the reference within ``1e-4 * (1 + |opt|)``;
  (b) no dual bound past the reference by more than ``1e-6 * (1 + |opt|)``;
  (c) no incumbent better than the reference by more than ``1e-6 * (1 + |opt|)``;
  (d) every returned point has ``max_viol <= 1e-6`` on the original rows/box/integrality.

Population: ``p42_rens_panel.population()`` (MIPLIB-38 + HiGHS ``check`` panel M), on the
Rust MILP route (``DISCOPT_LP_MILP_BACKEND=rust``), deterministic with a node budget so
node counts compare the same search across builds rather than two loads of the machine.

Usage (one run per build; compare the two JSONL files with ``compare``)::

    python -u pseudocost_sentinel_panel.py run --out before.jsonl
    python -u pseudocost_sentinel_panel.py run --out after.jsonl
    python -u pseudocost_sentinel_panel.py compare before.jsonl after.jsonl

Discipline (CLAUDE.md §6-§10): per-solve progress; exceptions propagate; the loaded
``discopt`` is asserted to be the tree under test and the extension's sha256 is recorded
per row; the run exits non-zero when no check executed or no pseudocost update was
observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
PANEL_PY = HERE.parent / "lp_milp_panel.py"
sys.path.insert(0, str(HERE.parent))

import p42_rens_panel as P42  # noqa: E402, N812

_CTR = re.compile(r"^\s+(Pseudocost\w*)\s+(\d+)$")
SO = REPO / "python" / "discopt" / "_rust.abi3.so"


def so_sha() -> str:
    return hashlib.sha256(SO.read_bytes()).hexdigest()[:16]


def solve(path: str, a) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_LP_STD_ENGINE"}
    env.update(
        PYTHONUNBUFFERED="1",
        DISCOPT_LP_MILP_BACKEND="rust",
        DISCOPT_PROFILE="1",
        DISCOPT_PANEL_DETERMINISTIC="1",
    )
    cmd = [sys.executable, str(PANEL_PY), "worker", path, "R", str(a.time_limit),
           str(a.gap), str(a.max_nodes)]  # fmt: skip
    load = os.getloadavg()[0]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3 * a.time_limit + 180)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if p.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"{path}: worker rc={p.returncode}\n{p.stderr[-2000:]}")
    res = json.loads(lines[0][len("RESULT ") :])
    ctr: dict[str, int] = {}
    for ln in p.stderr.splitlines():
        mm = _CTR.match(ln)
        if mm:
            ctr[mm.group(1)] = ctr.get(mm.group(1), 0) + int(mm.group(2))
    res["pc"] = ctr
    res["load"] = load
    return res


def check(res: dict, opt: float, maximize: bool) -> tuple[int, list[str]]:
    """Return (executed checks, violations) for one result against the reference."""
    sgn = -1.0 if maximize else 1.0
    n, bad = 0, []
    inc, bnd = res.get("objective"), res.get("bound")
    if res.get("gap_certified"):
        n += 1
        if inc is None or abs(inc - opt) > 1e-4 * (1 + abs(opt)):
            bad.append(f"certified objective {inc} != opt {opt}")
    if bnd is not None and math.isfinite(bnd):
        n += 1
        if sgn * (bnd - opt) > 1e-6 * (1 + abs(opt)):
            bad.append(f"bound {bnd} past opt {opt}")
    if inc is not None:
        n += 1
        if sgn * (opt - inc) > 1e-6 * (1 + abs(opt)):
            bad.append(f"incumbent {inc} better than opt {opt}")
    if res.get("max_viol") is not None:
        n += 1
        if res["max_viol"] > 1e-6:
            bad.append(f"max_viol {res['max_viol']}")
    return n, bad


def run(a) -> int:
    import discopt

    if not discopt.__file__.startswith(str(REPO / "python")):
        raise RuntimeError(f"loaded discopt from {discopt.__file__}")
    sha = so_sha()
    pop = P42.population()
    if a.only:
        keep = set(a.only.split(","))
        pop = [m for m in pop if m["name"] in keep]
    print(f"build {sha}  {len(pop)} instances  max_nodes={a.max_nodes}", flush=True)
    executed, violations, updates = 0, 0, 0
    with open(a.out, "w") as fh, ThreadPoolExecutor(a.jobs) as ex:
        futs = [(m, ex.submit(solve, str(REPO / m["path"]), a)) for m in pop]
        for m, f in futs:
            res = f.result()
            n, bad = check(res, m["oracle_objective"], m["maximize"])
            executed += n
            violations += len(bad)
            pc = res["pc"]
            updates += pc.get("PseudocostUpdates", 0)
            row = {"name": m["name"], "panel": m["panel"], "so": sha, "opt": m["oracle_objective"],
                   "violations": bad, **res}  # fmt: skip
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(
                f"{m['name']:28s} {res['status']:12s} cert={int(res['gap_certified'])} "
                f"nodes={res['nodes']:7d} upd={pc.get('PseudocostUpdates', 0):7d} "
                f"sent={pc.get('PseudocostSentinelSkipped', 0):6d} "
                f"untr={pc.get('PseudocostUntrustedChild', 0):5d} "
                f"load={res['load']:.1f} {'VIOL ' + '; '.join(bad) if bad else ''}",
                flush=True,
            )
    print(f"executed checks: {executed}  violations: {violations}  pseudocost updates: {updates}")
    if executed == 0 or updates == 0:
        print("probe measured nothing", file=sys.stderr)
        return 2
    return 1 if violations else 0


def compare(a) -> int:
    rows = [
        {r["name"]: r for r in map(json.loads, Path(p).read_text().splitlines())}
        for p in (a.before, a.after)
    ]
    b, c = rows
    shas = {r["so"] for r in b.values()}, {r["so"] for r in c.values()}
    if len(shas[0]) != 1 or len(shas[1]) != 1 or shas[0] == shas[1]:
        raise RuntimeError(f"build shas not one-per-file and distinct: {shas}")
    names = sorted(set(b) & set(c))
    n = 0
    fired = [x for x in names if b[x]["pc"].get("PseudocostSentinelSkipped", 0) > 0]
    print(f"{'instance':28s} {'sent':>6s} {'nodes0':>8s} {'nodes1':>8s} cert0 cert1  obj0/obj1")
    for x in names:
        r0, r1 = b[x], c[x]
        n += 1
        mark = "" if r0["nodes"] == r1["nodes"] else " *"
        print(
            f"{x:28s} {r0['pc'].get('PseudocostSentinelSkipped', 0):6d} {r0['nodes']:8d} "
            f"{r1['nodes']:8d} {int(r0['gap_certified']):5d} {int(r1['gap_certified']):5d}  "
            f"{r0['objective']} / {r1['objective']}{mark}"
        )
    cert = [sum(int(r[x]["gap_certified"]) for x in names) for r in rows]
    viol = [sum(len(r[x]["violations"]) for x in names) for r in rows]
    moved = [x for x in names if b[x]["nodes"] != c[x]["nodes"]]
    print(f"\ncompared {n}; sentinel fired on {len(fired)}: {fired}")
    print(f"certified before/after: {cert[0]}/{cert[1]}; violations before/after: {viol}")
    print(f"node count changed on {len(moved)}: {moved}")
    lost = [x for x in names if b[x]["gap_certified"] and not c[x]["gap_certified"]]
    gained = [x for x in names if c[x]["gap_certified"] and not b[x]["gap_certified"]]
    print(f"CERT_LOST={lost} CERT_GAINED={gained}")
    for grp, sel in (("fired", fired), ("all", names)):
        both = [x for x in sel if b[x]["gap_certified"] and c[x]["gap_certified"]]
        print(
            f"nodes on {grp} instances certified in both ({len(both)}): "
            f"{sum(b[x]['nodes'] for x in both)} -> {sum(c[x]['nodes'] for x in both)}"
        )
    return 0 if n else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--time-limit", type=float, default=120.0)
    r.add_argument("--gap", type=float, default=P42.P.GAP)
    r.add_argument("--max-nodes", type=int, default=5000)
    r.add_argument("--jobs", type=int, default=3)
    r.add_argument("--only", default="")
    c = sub.add_parser("compare")
    c.add_argument("before")
    c.add_argument("after")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
