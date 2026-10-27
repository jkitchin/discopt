#!/usr/bin/env python3
"""#1995: CLAUDE.md section 5 panel for ``DISCOPT_KERNEL_UNDECIDED_RETRY``, GDP class.

The MINLPLib panel (``issue1944_rlt_panel.py --with-corpus``) holds no disjunction,
so it cannot reach the ``gdp.simplex`` lowerings #1995 is about. This panel adds:

* every case of the serialize bound-neutral panel (``serialize_builder_cases.
  ROUND_TRIPS`` minus ``test_serialize_builders.NOT_SOLVED_HERE``: all areas --
  dae, ml, gdp, stochastic, ro, bilevel, gp and every registered transformation);
* the three native GDPlib models (``gdplib_native.NATIVE_BUILDERS``) under
  ``gdp_method="simplex"``, against their BARON/SCIP-confirmed optima.

No name is hard-coded. Each solve runs in its own subprocess and the arm order
alternates per case (interleaved A/B, section 9). Checks, per arm:

* cert-clean -- no CRASH; a certified objective equals the oracle (native GDPlib)
  or the other arm's certified objective; no bound is above any verified
  incumbent of either arm or the oracle (by sense); every incumbent passes
  ``verify_point`` on a freshly built model (a model ``verify_point`` cannot read
  is reported as such, never as verified); no case certified OFF loses it ON;
* net-positive -- certificates gained, nodes and wall where both certify.

``--report`` exits non-zero when nothing was compared, a check failed, or a
certificate was lost. The ON arm reports ``tree/prop_retries`` (section 6).

    python -u discopt_benchmarks/scripts/issue1995_gdp_panel.py --out <dir> --time-limit 30
    python discopt_benchmarks/scripts/issue1995_gdp_panel.py --out <dir> --time-limit 30 --report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FLAG = "DISCOPT_KERNEL_UNDECIDED_RETRY"
TOL = 1e-4


def _cases() -> dict:
    sys.path.insert(0, str(REPO / "python" / "tests"))
    sys.path.insert(0, str(REPO / "discopt_benchmarks"))
    import serialize_builder_cases as cases
    from test_serialize_builders import NOT_SOLVED_HERE

    from benchmarks.gdplib_native import NATIVE_BUILDERS

    out = {f"serialize:{k}": (fn, {}) for k, fn in cases.ROUND_TRIPS.items()}
    for k in NOT_SOLVED_HERE:
        out.pop(f"serialize:{k}")
    for name, fn in NATIVE_BUILDERS.items():
        out[f"gdplib:{name}:simplex"] = (fn, {"gdp_method": "simplex"})
    return out


def _oracle(key: str):
    if not key.startswith("gdplib:"):
        return None
    sys.path.insert(0, str(REPO / "discopt_benchmarks"))
    from benchmarks.gdplib_runner import reference_optima

    return reference_optima().get(key.split(":")[1])


def _solve_one(key: str, tl: float) -> None:
    import discopt
    import numpy as np
    from discopt.validation.feasibility import verify_point

    assert discopt.__file__.startswith(str(REPO)), discopt.__file__
    fn, kwargs = _cases()[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fn()
        t0 = time.perf_counter()
        try:
            r = model.solve(time_limit=tl, **kwargs)
        except NotImplementedError as exc:  # a semantics the solver declines (#1124)
            print("RESULT " + json.dumps({"status": "raised", "err": str(exc)[:200]}))
            return
        wall = time.perf_counter() - t0
        verified = None
        verify_error = None
        if r.x is not None and r.objective is not None:
            fresh = fn()
            try:
                flat = np.concatenate(
                    [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
                )
                verified = bool(verify_point(fresh, flat).ok)
            except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
                verify_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    from discopt.modeling.core import ObjectiveSense

    obj = model._objective
    sense = "max" if obj is not None and obj.sense == ObjectiveSense.MAXIMIZE else "min"
    ss = r.solver_stats or {}
    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(r.status),
                "gap_certified": bool(r.gap_certified),
                "objective": f(r.objective),
                "bound": f(r.bound),
                "sense": sense,
                "nodes": r.node_count,
                "wall": wall,
                "retries": ss.get("tree/prop_retries"),
                "retry_decided": ss.get("tree/prop_retry_decided"),
                "verified": verified,
                "verify_error": verify_error,
            }
        ),
        flush=True,
    )


def run(out: Path, tl: float) -> Path:
    results = out / f"gdp_tl{int(tl)}.jsonl"
    done = set()
    if results.exists():
        for ln in results.open():
            j = json.loads(ln)
            done.add((j["key"], j["arm"]))
    keys = sorted(_cases())
    print(f"population {len(keys)}", flush=True)
    executed = 0
    with results.open("a") as fh:
        for idx, key in enumerate(keys):
            for arm in ("1", "0") if idx % 2 else ("0", "1"):
                if (key, arm) in done:
                    continue
                env = dict(os.environ, **{FLAG: arm})
                t0 = time.perf_counter()
                try:
                    proc = subprocess.run(
                        [sys.executable, __file__, "--_solve", key, str(tl)],
                        capture_output=True,
                        text=True,
                        timeout=tl + 180,
                        env=env,
                    )
                    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
                    res = (
                        json.loads(lines[-1][7:])
                        if lines
                        else {"status": "CRASH", "rc": proc.returncode, "err": proc.stderr[-1500:]}
                    )
                except subprocess.TimeoutExpired:
                    res = {"status": "HARD_TIMEOUT"}
                res.update(key=key, arm=arm, opt=_oracle(key), proc_wall=time.perf_counter() - t0)
                executed += 1
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                print(
                    f"[{executed}] {key:48s} arm={arm} {res['status']:12s} "
                    f"obj={res.get('objective')} bound={res.get('bound')} "
                    f"nodes={res.get('nodes')} retries={res.get('retries')} "
                    f"t={res['proc_wall']:.1f}",
                    flush=True,
                )
    print(f"EXECUTED {executed}", flush=True)
    return results


def _close(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-6 + TOL * max(1.0, abs(b))


def report(path: Path) -> int:
    by: dict[str, dict] = {}
    for ln in path.open():
        j = json.loads(ln)
        by.setdefault(j["key"], {})[j["arm"]] = j
    checks = problems = fired = unverifiable = 0
    gained, lost = [], []
    both_nodes, both_wall, total_wall = [0, 0], [0.0, 0.0], [0.0, 0.0]
    for key, arms in sorted(by.items()):
        if set(arms) != {"0", "1"}:
            continue
        off, on = arms["0"], arms["1"]
        incumbents = [
            r["objective"]
            for r in (off, on)
            if r.get("verified") and r.get("objective") is not None
        ]
        for row in (off, on):
            if row["status"] == "raised":
                continue
            checks += 4
            tag = f"{key} arm={row['arm']}"
            if row["status"] in ("CRASH", "HARD_TIMEOUT"):
                problems += 1
                print(f"PROBLEM {tag} {row['status']} {row.get('err', '')[-300:]}")
                continue
            other = on if row is off else off
            if row.get("gap_certified") and row.get("status") == "optimal":
                ref = row["opt"] if row.get("opt") is not None else None
                if ref is None and other.get("gap_certified"):
                    ref = other.get("objective")
                if ref is not None and not _close(row.get("objective"), ref):
                    problems += 1
                    print(f"PROBLEM {tag} certified {row.get('objective')} != {ref}")
            b = row.get("bound")
            if b is not None:
                for v in incumbents + ([row["opt"]] if row.get("opt") is not None else []):
                    slack = 1e-6 + TOL * max(1.0, abs(v))
                    if (b < v - slack) if row.get("sense") == "max" else (b > v + slack):
                        problems += 1
                        print(f"PROBLEM {tag} bound {b} past a feasible value {v}")
            if row.get("verified") is False:
                problems += 1
                print(f"PROBLEM {tag} incumbent fails verify_point")
            if row.get("verify_error"):
                unverifiable += 1
                print(f"  unverifiable {tag}: {row['verify_error']}")
        total_wall[0] += off.get("wall") or 0.0
        total_wall[1] += on.get("wall") or 0.0
        if (on.get("retries") or 0) > 0:
            fired += 1
            print(
                f"  fired        {key:48s} retries={on.get('retries')} "
                f"decided={on.get('retry_decided')}  {off['status']} -> {on['status']}"
            )
        c0, c1 = bool(off.get("gap_certified")), bool(on.get("gap_certified"))
        if c0 and not c1:
            lost.append(key)
        if c1 and not c0:
            gained.append(key)
        if c0 and c1:
            both_nodes[0] += off.get("nodes") or 0
            both_nodes[1] += on.get("nodes") or 0
            both_wall[0] += off["wall"]
            both_wall[1] += on["wall"]
            if off.get("nodes") != on.get("nodes"):
                print(f"  both certify {key:48s} nodes {off.get('nodes')} -> {on.get('nodes')}")
    print(f"cases {len(by)}, retry fired on {fired}, unverifiable incumbents {unverifiable}")
    print(f"certified gained {gained}")
    print(f"certified lost   {lost}")
    print(
        f"both certify: nodes {both_nodes[0]} -> {both_nodes[1]}, "
        f"wall {both_wall[0]:.1f} -> {both_wall[1]:.1f} s"
    )
    print(f"total wall {total_wall[0]:.1f} -> {total_wall[1]:.1f} s")
    print(f"executed checks {checks}, problems {problems}")
    return 1 if checks == 0 or problems or lost else 0


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        _solve_one(sys.argv[2], float(sys.argv[3]))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        sys.exit(report(a.out / f"gdp_tl{int(a.time_limit)}.jsonl"))
    a.out.mkdir(parents=True, exist_ok=True)
    run(a.out, a.time_limit)


if __name__ == "__main__":
    main()
