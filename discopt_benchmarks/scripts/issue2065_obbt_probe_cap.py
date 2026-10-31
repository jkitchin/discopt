#!/usr/bin/env python3
"""#2065 entry experiment: do persistent OBBT probes overrun ``time_limit_per_lp``?

``run_obbt_on_relaxation(time_limit_per_lp=...)`` documents a per-LP limit that the
persistent probe path (``_PersistentProbeLP.solve``) does not forward; since #2006
each probe is capped only by the sweep's remaining ``deadline``. Before honoring the
per-LP cap (a bound-changing change, CLAUDE.md section 5) this measures, per probe on
the default solve path:

* its wall time, and the caller's ``time_limit_per_lp`` (read from the
  ``_solve_probe`` closure that called the probe -- the exact value the caller
  passed, not a guess);
* whether its optimum would tighten the box (the raw vertex vs the current bound,
  with the sweep's ``eps``; an upper bound on what the NS-clamped sweep tightens).

A probe "exceeds the cap" when its wall time is above the caller's
``time_limit_per_lp``. Kill criterion (from the issue): under ~1% of probes exceed
the cap, or none that do tighten -> record and document; otherwise run the panel.

Each instance runs in its own subprocess (hard timeout ``time_limit + 60`` s) and
appends one JSON line to ``<out>/probes.jsonl``; a rerun resumes. Every instance
line carries its probe count, and ``--analyze`` exits non-zero when no probe fired
anywhere (CLAUDE.md section 6).

Usage::

    python -u discopt_benchmarks/scripts/issue2065_obbt_probe_cap.py --out <dir> \\
        --corpus inrepo --time-limit 30 --workers 3
    python -u discopt_benchmarks/scripts/issue2065_obbt_probe_cap.py --out <dir> \\
        --corpus minlplib200 --time-limit 30 --workers 3
    python -u discopt_benchmarks/scripts/issue2065_obbt_probe_cap.py --out <dir> \\
        --files <dir>/QPLIB_5875.qplib --time-limit 30 --workers 1
    python discopt_benchmarks/scripts/issue2065_obbt_probe_cap.py --out <dir> --analyze
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INREPO = REPO / "python" / "tests" / "data" / "minlplib_nl"
EPS = 1e-7  # run_obbt_on_relaxation's default ``eps``


def _child(nl: str, tl: float) -> None:
    """Subprocess entry: solve ``nl`` with the probe recorder installed."""
    import discopt
    import discopt.modeling as dm
    import numpy as np
    from discopt._relax import obbt

    # Section 8: confirm which code is loaded and that it carries #2006's deadline
    # forwarding (the baseline this measurement is about).
    assert "time_limit_s=budget" in Path(obbt.__file__).read_text(), obbt.__file__

    probes: list[list] = []
    orig = obbt._PersistentProbeLP.solve

    def rec(self, c, lb_arr, ub_arr, warm_basis, time_limit_s=None):
        f = sys._getframe(1)  # ``_solve_probe`` inside ``run_obbt_on_relaxation``
        cap = f.f_locals["time_limit_per_lp"]
        out = orig(self, c, lb_arr, ub_arr, warm_basis, time_limit_s=time_limit_s)
        st, obj, _d, _b, wall = out
        j = int(np.argmax(np.abs(c)))
        tight = False
        if obj is not None and st == obbt.SolveStatus.OPTIMAL:
            tight = bool(obj > lb_arr[j] + EPS) if c[j] > 0 else bool(-obj < ub_arr[j] - EPS)
        probes.append([wall, cap, time_limit_s, str(st.value), tight])
        return out

    obbt._PersistentProbeLP.solve = rec
    t0 = time.perf_counter()
    if nl.endswith(".qplib"):
        from discopt.interfaces import qplib

        model = qplib.from_qplib(nl)
    else:
        model = dm.from_nl(nl)
    r = model.solve(time_limit=tl, gap_tolerance=1e-4)
    out = {
        "discopt_file": discopt.__file__,
        "status": str(r.status),
        "objective": r.objective,
        "bound": r.bound,
        "wall": time.perf_counter() - t0,
        "probes": probes,
    }
    print("RESULT " + json.dumps(out), flush=True)


def _panel(corpus: str, out: Path, files: list[str]) -> list[tuple[str, Path]]:
    if files:
        return [(Path(f).stem, Path(f)) for f in files]
    if corpus == "inrepo":
        return [(p.stem, p) for p in sorted(INREPO.glob("*.nl"))]
    sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))
    import scip_wins_screen as s

    rows = s.build_panel(out, 200, 200, None)
    res = []
    for r in rows:
        dest = out / "nl" / f"{r['name']}.nl"
        res.append((r["name"], s._fetch(f"{s.MINLPLIB}/nl/{r['name']}.nl", dest)))
    return res


def run(out: Path, corpus: str, tl: float, workers: int, files: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    log = out / "probes.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(ln)["name"] for ln in log.read_text().splitlines() if ln.strip()}
    jobs = [(n, p) for n, p in _panel(corpus, out, files) if n not in done]
    print(f"{len(jobs)} instances to run ({len(done)} done)", flush=True)

    def one(job):
        name, path = job
        t0 = time.perf_counter()
        rec = {"name": name, "corpus": corpus}
        try:
            cp = subprocess.run(
                [sys.executable, "-u", __file__, "--child", str(path), "--time-limit", str(tl)],
                capture_output=True,
                text=True,
                timeout=tl + 60,
            )
            line = next((ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if line is None:
                rec.update(error="no result", stderr=cp.stderr[-2000:])
            else:
                rec.update(json.loads(line[7:]))
        except subprocess.TimeoutExpired:
            rec.update(error="timeout")
        rec["harness_wall"] = time.perf_counter() - t0
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        n = len(rec.get("probes", []))
        print(f"{name}: {n} probes, {rec.get('error', rec.get('status'))}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


def analyze(out: Path) -> None:
    rows = [json.loads(ln) for ln in (out / "probes.jsonl").read_text().splitlines() if ln]
    n_inst = len(rows)
    errors = [r["name"] for r in rows if "error" in r]
    n_probe = n_capped = n_over = n_over_tight = n_tight = 0
    over_wall = 0.0
    per_inst: dict[str, list] = {}
    by_cap: dict[float, list[int]] = {}
    for r in rows:
        for wall, cap, _tls, _st, tight, *_ in r.get("probes", []):
            n_probe += 1
            n_tight += tight
            if cap is None:
                continue
            n_capped += 1
            c = by_cap.setdefault(cap, [0, 0, 0])
            c[0] += 1
            if wall > cap:
                n_over += 1
                n_over_tight += tight
                over_wall += wall - cap
                c[1] += 1
                c[2] += tight
                per_inst.setdefault(r["name"], [0, 0, 0.0])
                per_inst[r["name"]][0] += 1
                per_inst[r["name"]][1] += tight
                per_inst[r["name"]][2] = max(per_inst[r["name"]][2], wall)
    inst_with_probes = sum(1 for r in rows if r.get("probes"))
    print(f"instances: {n_inst} ({inst_with_probes} ran a persistent probe; errors {errors})")
    print(f"probes executed: {n_probe} (tightening: {n_tight}); with a per-LP cap: {n_capped}")
    pct = 100.0 * n_over / max(n_capped, 1)
    print(f"probes over the cap: {n_over} ({pct:.3f}%), of which tightened: {n_over_tight}")
    print(f"wall spent past the cap: {over_wall:.2f} s")
    print("by time_limit_per_lp: [probes, over cap, over cap & tightened]")
    for k, v in sorted(by_cap.items()):
        print(f"  {k}: {v}")
    print("instances with over-cap probes: name [over, over&tight, max probe wall s]")
    for k, v in sorted(per_inst.items(), key=lambda kv: -kv[1][2]):
        print(f"  {k}: [{v[0]}, {v[1]}, {v[2]:.3f}]")
    if n_probe == 0:
        sys.exit("ERROR: no probe fired -- the instrument measured nothing")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--corpus", choices=["inrepo", "minlplib200"], default="inrepo")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--files", nargs="*", default=[], help=".nl/.qplib paths (overrides --corpus)")
    ap.add_argument("--child")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.child:
        _child(a.child, a.time_limit)
    elif a.analyze:
        analyze(a.out)
    else:
        run(a.out, a.corpus, a.time_limit, a.workers, a.files)


if __name__ == "__main__":
    main()
