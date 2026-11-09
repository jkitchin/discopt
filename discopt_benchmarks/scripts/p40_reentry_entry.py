"""highs-parity P4.0 entry: does spending run 1's leftover on a re-entry help?

On the MIPLIB-38 rows the Rust MILP route leaves uncertified (the P4.1 set, from the
``--baseline`` run), solve each on that route (``DISCOPT_LP_MILP_BACKEND=rust``, fresh
subprocess per solve) at ``--time-limit`` with ``DISCOPT_MILP_REENTRY_SPEND`` OFF and ON,
interleaved with the arm order rotated per instance and repetition. Each row is classified
against the published optimum as in ``miplib_panel.classify``, and the ON row is compared
with the OFF row on the same repetition: ``primal+`` / ``dual+`` when ON is strictly
better by more than ``--eps`` (relative to ``1 + |opt|``), ``primal-`` / ``dual-`` when
it is worse.

The flag only changes when the #698 re-entry runs, so the probe proves it fired by wall:
an ON row that ran past ``0.75 * time_limit`` where its OFF twin stopped before
``0.6 * time_limit``. The run prints executed / fired counts and exits non-zero when
either is zero (CLAUDE.md section 6). Exceptions propagate; progress is per solve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))

import lp_milp_loader as L  # noqa: E402, N812
import lp_milp_panel as P  # noqa: E402, N812
import miplib_panel as M  # noqa: E402, N812

FLAG = "DISCOPT_MILP_REENTRY_SPEND"


def _solve(path: str, on: bool, a: argparse.Namespace) -> dict:
    if on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)
    try:
        ns = argparse.Namespace(time_limit=a.time_limit, gap=a.gap, max_nodes=a.max_nodes)
        load = os.getloadavg()[0]
        res = P._run_sub(path, "R", ns)
    finally:
        os.environ.pop(FLAG, None)
    if "error" in res:
        raise RuntimeError(f"{path} on={on}: {res}")
    res["load"] = load
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="docs/dev/data/highs-parity/baseline-dcdac02/miplib.json")
    ap.add_argument("--panel-json", default=str(M.PANEL_JSON))
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=P.GAP)
    ap.add_argument("--max-nodes", type=int, default=100_000_000)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--instances", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    os.environ.pop(FLAG, None)
    P._assert_discopt_under_test()
    hdr = P._header()
    base = json.loads(Path(a.baseline).read_text())
    names = [r["name"] for r in base["rows"] if not r["R"]["gap_certified"]]
    if a.instances:
        keep = set(a.instances.split(","))
        names = [n for n in names if n in keep]
    man = {m["name"]: m for m in M._load_panel(Path(a.panel_json))["members"]}
    print("header", json.dumps(hdr), "instances", len(names), flush=True)

    rows, executed, fired = [], 0, 0
    tally: dict[str, int] = {}
    for i, name in enumerate(names):
        mem = man[name]
        path = str(REPO / mem["path"])
        inst = L.read_instance(path)
        opt = mem["oracle_objective"]
        for rep in range(a.reps):
            order = (False, True) if (i + rep) % 2 == 0 else (True, False)
            got = {}
            for on in order:
                res = _solve(path, on, a)
                P.check_route(name, "R", res)
                executed += 1
                got[on] = {**M.classify(res, opt, inst.maximize, a.gap),
                           **{k: res.get(k) for k in ("status", "objective", "bound", "nodes",
                                                      "wall", "termination", "load")}}  # fmt: skip
            off, on_ = got[False], got[True]
            did_fire = (on_["wall"] or 0) > 0.75 * a.time_limit and (
                off["wall"] or 0
            ) < 0.6 * a.time_limit
            fired += did_fire
            verdict = []
            for side in ("primal", "dual"):
                x0, x1 = off[f"{side}_short"], on_[f"{side}_short"]
                if x0 is None and x1 is not None:
                    verdict.append(f"{side}+")
                elif x1 is None and x0 is not None:
                    verdict.append(f"{side}-")
                elif x0 is not None and x1 is not None:
                    if x1 < x0 - a.eps:
                        verdict.append(f"{side}+")
                    elif x1 > x0 + a.eps:
                        verdict.append(f"{side}-")
            if on_["class"] == "solved" and off["class"] != "solved":
                verdict.append("certified+")
            if off["class"] == "solved" and on_["class"] != "solved":
                verdict.append("certified-")
            for v in verdict or ["tie"]:
                tally[v] = tally.get(v, 0) + 1
            rows.append({"name": name, "rep": rep, "opt": opt, "maximize": inst.maximize,
                         "off": off, "on": on_, "fired": did_fire, "verdict": verdict})  # fmt: skip
            print(f"[{i + 1}/{len(names)} r{rep}] {name:22s} fired={did_fire} "
                  f"off=({off['class']},{off['primal_short']},{off['dual_short']},"
                  f"{off['wall']:.1f}s) on=({on_['class']},{on_['primal_short']},"
                  f"{on_['dual_short']},{on_['wall']:.1f}s) {verdict or ['tie']}",
                  flush=True)  # fmt: skip
            Path(a.out).write_text(json.dumps({"header": hdr, "args": vars(a), "rows": rows,
                                               "tally": tally}, indent=1) + "\n")  # fmt: skip
    print(f"p40 entry: executed={executed} fired={fired} tally={tally}", flush=True)
    return 0 if executed and fired else 1


if __name__ == "__main__":
    sys.exit(main())
