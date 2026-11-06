"""highs-parity P4.4 entry: the Rust MILP route's ROOT dual gap at HEAD.

Each instance is solved through ``Model.solve`` on the Rust route
(``DISCOPT_LP_MILP_BACKEND=rust``, ``max_nodes=2``) in a fresh subprocess, with
``discopt._rust.solve_milp_csc_py`` wrapped to record what the FIRST driver call returns:
that call carries exactly the options the shipped route builds (root cut loop included),
and its bound is the driver's root bound. ``max_nodes=2``, not 1: at 1 only the root is
*created*, the tree's global bound is still ``-inf`` (measured on ``mik-250-20-75-1``);
at 2 the root is solved and branched, so the global bound is the root's (identical at 2
and 3: -51756.66). The solve's *reported* bound is not used: with
no incumbent a node-capped run defers to a fallback engine, whose bound is reported
instead (measured: ``sp150x300d`` read 91.6 %, worse than its own LP). The residual is
``max(0, sgn*(opt - bound)) / (1 + |opt|)`` against the published optimum, the
milp-competitiveness-plan section 2.5 metric. Extra environment for an arm is passed with
``--env KEY=VALUE`` (repeatable), so a candidate separator's flag can be A/B'd on the
same probe; ``--all`` takes the MIPLIB-38 + M population of ``p42_rens_panel``;
``--profile`` sets ``DISCOPT_PROFILE=1`` and records the FIRST driver call's cut counters
(``RootCuts*``, ``SubstDrop*``, ``SepGomory*``; the first dump of each name is that call's,
since the profile prints after every driver call); ``--force-kw '{"cmir_cuts": true}'``
overrides keywords on the route's first
driver call only (an experiment knob for options the route does not expose). Prints
executed count; exits non-zero when nothing ran (CLAUDE.md section 6).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import subprocess  # noqa: E402

import lp_milp_loader as L  # noqa: E402, N812
import lp_milp_panel as P  # noqa: E402, N812
import miplib_panel as M  # noqa: E402, N812

CHILD = r"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
from discopt import _rust
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); m = from_nl(nl)
calls = []
real = _rust.solve_milp_csc_py
FORCE = json.loads(os.environ.get("P44_FORCE_KW", "{}"))
def spy(*a, **kw):
    if not calls:  # the route's root call only
        kw.update(FORCE)
    out = real(*a, **kw)
    calls.append({"status": out[0], "obj": float(out[2]), "bound": float(out[3]),
                  "nodes": int(out[4]), "obj_const": float(a[11]), "max_nodes": int(a[12])})
    return out
_rust.solve_milp_csc_py = spy
r = m.solve(time_limit=float(sys.argv[3]), gap_tolerance=float(sys.argv[4]), max_nodes=2)
print("RESULT " + json.dumps({"calls": calls, "reported_bound": r.bound,
                              "route": r.algorithm_route}))
"""

_CTR = re.compile(r"^\s+((?:RootCut|SubstDrop|SepGomory|SepCmir|SepCover)\w*)\s+(\d+)$")
DEFAULT = "mik-250-20-75-1,mik-250-20-75-2,mik-250-20-75-3,mik-250-20-75-5,fiber,sp150x300d"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default=DEFAULT)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--gap", type=float, default=P.GAP)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--force-kw", default="{}")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    extra = dict(kv.split("=", 1) for kv in a.env)
    P._assert_discopt_under_test()
    man = {m["name"]: m for m in M._load_panel(M.PANEL_JSON)["members"]}
    if a.all:
        import p42_rens_panel as R  # noqa: N812

        pop = R.population()
        man = {m["name"]: m for m in pop}
        a.instances = ",".join(dict.fromkeys(m["name"] for m in pop))
    rows, executed = [], 0
    saved = {k: os.environ.get(k) for k in extra}
    os.environ.update(extra)
    try:
        for name in a.instances.split(","):
            mem = man[name]
            path = str(REPO / mem["path"])
            inst = L.read_instance(path)
            opt = mem["oracle_objective"]
            env = dict(
                os.environ,
                DISCOPT_LP_MILP_BACKEND="rust",
                PYTHONUNBUFFERED="1",
                P44_FORCE_KW=a.force_kw,
            )
            if a.profile:
                env["DISCOPT_PROFILE"] = "1"
            p = subprocess.run([sys.executable, "-c", CHILD, path, str(HERE.parent),
                                str(a.time_limit), str(a.gap)], env=env, capture_output=True,
                               text=True, timeout=4 * a.time_limit + 120)  # fmt: skip
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            if p.returncode != 0 or len(lines) != 1:
                raise RuntimeError(f"{name}: rc={p.returncode} {p.stderr[-1500:]}")
            res = json.loads(lines[0][len("RESULT ") :])
            if not res["calls"]:
                # Another route served it (e.g. the Python MILP tree): recorded and
                # counted, never silently dropped -- and it measures nothing here.
                rows.append({"name": name, "off_route": res["route"]})
                print(json.dumps(rows[-1]), flush=True)
                continue
            executed += 1
            c0 = res["calls"][0]
            ctr: dict[str, int] = {}
            for ln in p.stderr.splitlines():
                mm = _CTR.match(ln)
                if mm and mm.group(1) not in ctr:
                    ctr[mm.group(1)] = int(mm.group(2))
            # The driver minimises sgn * f; its bound includes obj_const.
            sgn = -1.0 if inst.maximize else 1.0
            gap = max(0.0, sgn * opt - c0["bound"]) / (1.0 + abs(opt))
            row = {"name": name, "opt": opt, "driver_bound": c0["bound"],
                   "root_gap_pct": 100 * gap, "driver_status": c0["status"],
                   "driver_nodes": c0["nodes"], "reported_bound": res["reported_bound"],
                   "route": res["route"], "env": extra,
                   "force_kw": json.loads(a.force_kw), "counters": ctr}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    Path(a.out).write_text(json.dumps({"header": P._header(), "rows": rows}, indent=1) + "\n")
    off = sum(1 for r in rows if "off_route" in r)
    print(f"p44 root gap: executed={executed} off_route={off}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
