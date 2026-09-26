"""Soundness + usefulness panel for ``dm.nonlinear_to_pwl(mode="outer")`` (#1482).

For every in-repo corpus instance (``python/tests/data/minlplib{,_nl}``) that has
a reference optimum in ``python/tests/data/known_optima.toml`` and at least one
univariate nonlinear term the transformation can replace:

* build the outer approximation and solve it with refinement;
* SOUNDNESS (hard): the reported bound never passes the reference optimum
  (``bound <= opt + tol`` for min, ``>=`` for max), and a ``gap_certified``
  result's objective matches the reference within the certification tolerance;
  with ``--mode approximate``: no bound, no certificate, never ``"optimal"``, and
  a reported (verified) objective never beats the reference optimum;
* report status, rounds, terms, and wall time next to a plain ``Model.solve``.

Prints one line per instance and an executed-check count; exits 1 on any
soundness violation, 2 if no check executed (CLAUDE.md §6).

Usage::

    python -u discopt_benchmarks/scripts/nonlinear_to_pwl_corpus_panel.py \
        [--time-limit 60] [--only name1,name2]
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

import discopt.modeling as dm
from discopt.modeling._pwl_transform import PWLTransformError, nonlinear_to_pwl

ROOT = Path(__file__).resolve().parents[2] / "python" / "tests" / "data"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--mode", choices=("outer", "approximate"), default="outer")
    ap.add_argument("--baseline", action="store_true", help="also time a plain Model.solve")
    args = ap.parse_args(argv)

    optima = tomllib.loads((ROOT / "known_optima.toml").read_text())
    files = {p.stem: p for d in ("minlplib", "minlplib_nl") for p in (ROOT / d).glob("*.nl")}
    names = sorted(n for n in optima if isinstance(optima[n], dict) and n in files)
    if args.only:
        wanted = set(args.only.split(","))
        names = [n for n in names if n in wanted]

    checks = 0
    violations = 0
    rows = []
    for name in names:
        opt = float(optima[name]["optimum"])
        m = dm.from_nl(str(files[name]))
        try:
            t = nonlinear_to_pwl(m, mode=args.mode, segments=args.segments)
        except PWLTransformError as exc:
            print(f"{name:22s} refused: {exc}", flush=True)
            continue
        if not t.terms:
            print(f"{name:22s} no transformable univariate term", flush=True)
            continue
        maximize = m._objective.sense.value == "maximize"
        t0 = time.perf_counter()
        r = t.solve(time_limit=args.time_limit)
        wall = time.perf_counter() - t0
        tol = 1e-4 * max(1.0, abs(opt)) + 1e-6
        ok = True
        if r.bound is not None:
            checks += 1
            if (r.bound > opt + tol) if not maximize else (r.bound < opt - tol):
                ok = False
        if r.gap_certified and r.objective is not None:
            checks += 1
            if abs(r.objective - opt) > 2 * tol:
                ok = False
        if r.status == "infeasible":
            ok = False  # every corpus instance with a known optimum is feasible
        if args.mode == "approximate":
            # The contract: never a bound, never a certificate, never "optimal".
            checks += 1
            if r.bound is not None or r.gap_certified or r.status == "optimal":
                ok = False
            if r.objective is not None:
                # A verified point of the original can never beat its optimum.
                checks += 1
                if (r.objective < opt - tol) if not maximize else (r.objective > opt + tol):
                    ok = False
        violations += not ok
        base = ""
        if args.baseline:
            t1 = time.perf_counter()
            rb = m.solve(time_limit=args.time_limit)
            base = f" | plain: {rb.status} {time.perf_counter() - t1:6.2f}s"
        st = r.solver_stats["nonlinear_to_pwl"]
        line = (
            f"{name:22s} {'OK ' if ok else 'BAD'} terms={len(t.terms):3d} "
            f"skipped={len(t.skipped):2d} milp={t.fully_linear!s:5s} "
            f"status={r.status:15s} cert={r.gap_certified!s:5s} rounds={st['rounds']:2d} "
            f"obj={r.objective!s:>22s} bound={r.bound!s:>22s} opt={opt:<14.8g} "
            f"wall={wall:6.2f}s{base}"
        )
        print(line, flush=True)
        rows.append(line)
    print(f"\nexecuted soundness checks: {checks}; violations: {violations}")
    if checks == 0:
        return 2
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
