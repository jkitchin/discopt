"""#2062: ``Model.solve()`` over the in-repo MINLPLib corpus, flag OFF vs ON, interleaved.

``corpus_scan.py`` compares the OA cut mask on the raw model; a solve may classify a
reformulated one, so this runs the solves themselves. Per instance: arm 0 then arm 1
(``DISCOPT_OA_CONVEXITY_TAYLOR``), each in a FRESH subprocess so no state leaks
between arms, at ``--time-limit`` seconds. Records status, objective, bound,
``gap_certified`` and ``node_count``, and checks:

* the certificate invariant and cross-arm agreement (no external oracle here;
  ``check_cert_neutrality.py`` covers the oracle-backed cert panel);
* the certificate invariant ``bound <= objective (+tol)`` (sign-adjusted for max);
* certification regressions (``gap_certified`` True OFF, False ON);
* exact ``node_count``/objective equality where both arms finish before the clock.

Prints a line per instance; exits non-zero if nothing ran (CLAUDE.md §6) or on any
soundness failure.

Usage: python -u corpus_solve_ab.py OUT.jsonl [--time-limit 30] [--shard k/n]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

FLAG = "DISCOPT_OA_CONVEXITY_TAYLOR"
ATOL, RTOL = 1e-6, 1e-4

_CHILD = r"""
import json, sys, time, math
import discopt._relax.convexity.certificate as cert
assert "_certify_univariate_taylor" in open(cert.__file__).read(), cert.__file__
from discopt.modeling.core import from_nl, ObjectiveSense
m = from_nl(sys.argv[1])
sense = m._objective.sense if m._objective is not None else ObjectiveSense.MINIMIZE
sign = 1.0 if sense == ObjectiveSense.MINIMIZE else -1.0
t0 = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2]))
wall = time.perf_counter() - t0
f = lambda v: None if v is None or not math.isfinite(float(v)) else float(v)
print("RESULT " + json.dumps(dict(status=str(r.status), objective=f(r.objective),
      bound=f(r.bound), gap_certified=bool(r.gap_certified),
      nodes=int(r.node_count or 0), wall=wall, sign=sign)))
"""


def run(path: str, arm: str, tl: float) -> dict:
    env = {**os.environ, FLAG: arm}
    try:
        p = subprocess.run(
            [sys.executable, "-c", _CHILD, path, str(tl)],
            env=env,
            capture_output=True,
            text=True,
            timeout=tl * 4 + 120,
        )
    except subprocess.TimeoutExpired:
        return {"error": "subprocess timeout"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"error": f"rc={p.returncode}: {p.stderr.strip().splitlines()[-1:]}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(v) for v in a.shard.split("/"))
    root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../../python/tests/data/minlplib_nl"
    )
    paths = sorted(glob.glob(os.path.join(root, "*.nl")))
    out = open(a.out, "w")
    ran = 0
    failures: list[str] = []
    identical = differ = clocked = 0
    for i, p in enumerate(paths):
        if i % n != k:
            continue
        name = os.path.basename(p)[:-3]
        r0, r1 = run(p, "0", a.time_limit), run(p, "1", a.time_limit)
        ran += 1
        row = {"model": name, "off": r0, "on": r1}
        out.write(json.dumps(row) + "\n")
        out.flush()
        for arm, r in (("off", r0), ("on", r1)):
            if "error" in r:
                continue
            sg = r["sign"]
            ob, bd = r["objective"], r["bound"]
            if (
                ob is not None
                and bd is not None
                and sg * bd > sg * ob + ATOL + RTOL * max(1, abs(ob))
            ):
                failures.append(f"{name} {arm}: bound {bd} > objective {ob}")
        if "error" in r0 or "error" in r1:
            if ("error" in r0) != ("error" in r1):
                failures.append(
                    f"{name}: error in one arm only: {r0.get('error')} / {r1.get('error')}"
                )
            tag = "ERROR"
        else:
            if r0["gap_certified"] and not r1["gap_certified"]:
                failures.append(f"{name}: certification lost ON")
            if r0["status"] == "optimal" and r1["status"] == "optimal":
                o0, o1 = r0["objective"], r1["objective"]
                if abs(o0 - o1) > ATOL + RTOL * max(1, abs(o0)):
                    failures.append(f"{name}: optimal objectives differ {o0} vs {o1}")
            settled = max(r0["wall"], r1["wall"]) < 0.9 * a.time_limit
            same = (r0["status"], r0["nodes"], r0["objective"]) == (
                r1["status"],
                r1["nodes"],
                r1["objective"],
            )
            if not settled:
                clocked += 1
                tag = "CLOCK"
            elif same:
                identical += 1
                tag = "SAME"
            else:
                differ += 1
                tag = "DIFF"
        print(f"{name}: {tag} off={r0} on={r1}", flush=True)
    print(
        f"ran {ran}; settled identical {identical}, settled differ {differ}, "
        f"hit clock {clocked}; failures {len(failures)}",
        flush=True,
    )
    for f in failures:
        print("  FAIL", f, flush=True)
    if ran == 0 or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
