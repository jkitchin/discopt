#!/usr/bin/env python3
"""#2013 census: which instances does the convex-MINLP route take?

For every instance of the ``scip_wins_screen`` population (MINLPLib ``.nl`` with a
proven status, ``nvars``/``ncons`` under the caps), one subprocess asks the router
(``solver._convex_minlp_auto_route``) for its decision. The route's master flags
(``DISCOPT_HIGHS_MASTER_LP_CUTS``) can only act on the routed set, so that set is
the population of their section 5 panel. Writes ``<out>/routed.txt``.

    python -u discopt_benchmarks/scripts/issue2013_route_census.py --out <dir> \\
        --max-vars 200 --max-cons 200 --workers 3

Prints per-instance progress and an EXECUTED count; exits non-zero when nothing
was censused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scip_wins_screen as screen  # noqa: E402


def _census_one(nl: str) -> None:
    import discopt.modeling as dm
    from discopt.solver import _convex_minlp_auto_route

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 50_000))
    method, reason, _ = _convex_minlp_auto_route(dm.from_nl(nl))
    print("RESULT " + json.dumps({"method": method, "reason": reason}), flush=True)
    os._exit(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-vars", type=int, default=200)
    ap.add_argument("--max-cons", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--_census", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._census:
        _census_one(a._census)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    panel = screen.build_panel(a.out, a.max_vars, a.max_cons, None)

    def one(p):
        nl = screen._fetch(f"{screen.MINLPLIB}/nl/{p['name']}.nl", a.out / "nl" / f"{p['name']}.nl")
        try:
            r = subprocess.run(
                [sys.executable, __file__, "--out", str(a.out), "--_census", str(nl)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
            res = json.loads(lines[-1][7:]) if lines else {"crash": r.stderr[-600:]}
        except subprocess.TimeoutExpired:
            res = {"timeout": True}
        res["name"] = p["name"]
        return res

    rows = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(one, panel), 1):
            rows.append(res)
            print(f"[{i}/{len(panel)}] {res['name']:28s} {res.get('method')}", flush=True)
    routed = sorted(r["name"] for r in rows if r.get("method"))
    errors = sorted(r["name"] for r in rows if "crash" in r or "timeout" in r)
    (a.out / "routed.txt").write_text("\n".join(routed) + "\n")
    print(f"routed {len(routed)}: {routed}")
    print(f"errors {len(errors)}: {errors}")
    print(f"EXECUTED {len(rows)}")
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
