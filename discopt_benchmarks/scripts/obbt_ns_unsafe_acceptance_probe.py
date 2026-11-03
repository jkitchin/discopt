#!/usr/bin/env python3
"""Does legacy (``DISCOPT_OBBT_NS_STRICT`` OFF) OBBT/DBBT cut off feasible points? (#1917)

Instrument only -- the solver code is not edited; every hook below wraps a module
global of ``discopt._relax.obbt`` and returns the wrapped function's value unchanged.

**Unsafe acceptance** (OFF path):

* OBBT ``_ns_select`` returns the raw LP vertex ``raw`` while the Neumaier-Shcherbina
  safe bound ``g`` exists and ``raw > g`` (a lower bound raised above ``g``, or an
  upper bound lowered below ``-g``); the *excess* is ``raw - g``. Kind ``a``.
* ``_ns_select`` returns ``raw`` because ``g`` is ``None``. Kind ``b``.
* Root DBBT (``dbbt_on_relaxation``): the legacy box is tighter than the strict box
  computed from the same relaxation (strict re-run with ``obbt_ns_strict=True``,
  result discarded). Kind ``dbbt``; excess = legacy bound minus strict bound.

Every acceptance is checked against a **reference feasible point** (MINLPLib ``.sol``,
QPLIB ``.sol``, or a solver optimum for GDPlib lowerings), itself verified with
:func:`discopt.warm_start.check_feasibility` (fail-closed). The OBBT cutoff is the
reference point's own internal objective plus ``1e-6 (1 + |f|)``, so the reference
point lies in every polytope probed. The final root box is also compared OFF vs ON.

DECISION RULE (fixed before any run, per the #1917 owner question):

* **DEFAULT-ON** iff any unsafe acceptance excludes a reference-feasible point by more
  than 1e-6 absolute, OR any instance's certified result (status, objective, bound vs
  oracle) differs between OFF and ON in a way that shows OFF wrong.
* Otherwise **keep OFF**, recording in the audit row the event frequency, the maximum
  excess and the number of reference points checked.

Usage::

    python -u discopt_benchmarks/scripts/obbt_ns_unsafe_acceptance_probe.py \\
        --corpus minlplib|qplib|gdplib [--max N] --out events.jsonl

Each instance runs in a fresh subprocess (``--child``). Ends with an executed-check
count and exits 2 when no OBBT probe was examined (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = Path(os.environ.get("DISCOPT_BENCH_DIR", "~/Dropbox/projects/discopt-minlp-benchmark"))
BENCH = BENCH.expanduser()
TOL_EXCL = 1e-6


# --------------------------------------------------------------------------- loading
def _load_minlplib(name):
    import discopt.modeling as dm
    import numpy as np

    nl = REPO / "python" / "tests" / "data" / "minlplib_nl" / f"{name}.nl"
    m = dm.from_nl(str(nl))
    colf = BENCH / "minlplib" / "nl" / f"{name}.col"
    cols = colf.read_text().split() if colf.exists() else []
    sol = BENCH / "minlplib" / "sol" / f"{name}.p1.sol"
    n = sum(int(v.size) for v in m._variables)
    if len(cols) != n or not sol.exists():
        return m, None, f"no reference (cols={len(cols)} n={n} sol={sol.exists()})"
    vals = {}
    for line in sol.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            vals[parts[0]] = float(parts[1])
    # MINLPLib .sol lists nonzeros only
    x = np.array([vals.get(c, 0.0) for c in cols], dtype=float)
    return m, x, ""


def _load_qplib(name):
    from discopt.interfaces import qplib

    inst = qplib.read_qplib(str(BENCH / "qplib" / "qplib" / f"{name}.qplib"))
    m = qplib.from_qplib(str(BENCH / "qplib" / "qplib" / f"{name}.qplib"))
    x, _ = qplib.read_solution(str(BENCH / "qplib" / "sol" / f"{name}.sol"), inst)
    return m, x, ""


def _load_gdplib(spec):
    import numpy as np
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.pyomo.gdp import convert_pyomo_gdp

    from benchmarks import gdplib_runner as gr

    inst, method = spec.rsplit(":", 1)
    m = convert_pyomo_gdp(gr.discover_models(include=[inst])[0].builder()).model
    ref = reformulate_gdp(m, method)
    r = ref.solve(time_limit=float(os.environ.get("PROBE_GDP_TL", "300")))
    if r.x is None:
        return ref, None, f"no reference: solve {r.status}"
    x = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in ref._variables])
    return ref, x, f"reference = discopt solve ({r.status}, cert={r.gap_certified})"


# --------------------------------------------------------------------------- child
def child(corpus: str, name: str) -> dict:
    import discopt
    import numpy as np
    from discopt import solver_tuning
    from discopt._relax import obbt as O  # noqa: N812
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.modeling.core import objective_sense_sign
    from discopt.warm_start import check_feasibility

    # §8: the code under test is the tree this script lives in.
    assert discopt.__file__ == str(REPO / "python" / "discopt" / "__init__.py"), discopt.__file__
    assert "def _ns_select(" in Path(O.__file__).read_text()  # #2077 marker
    assert not O._obbt_ns_strict(), "probe must run with DISCOPT_OBBT_NS_STRICT unset/OFF"

    out: dict = {"corpus": corpus, "name": name}
    loader = {"minlplib": _load_minlplib, "qplib": _load_qplib, "gdplib": _load_gdplib}[corpus]
    m, x, note = loader(name)
    out["ref_note"] = note
    lb0, ub0 = O._get_var_bounds(m)
    out["n"] = int(lb0.size)
    ref_ok = False
    cutoff = None
    if x is not None:
        ok, viol = check_feasibility(m, x, tol=1e-6)
        box_v = float(max(np.max(lb0 - x, initial=0.0), np.max(x - ub0, initial=0.0)))
        out.update(ref_feasible=bool(ok), ref_viol=viol[:3], ref_box_viol=box_v)
        ref_ok = bool(ok)
        f = float(make_evaluator(m).evaluate_objective(x))
        fi = float(objective_sense_sign(m)) * f
        cutoff = fi + 1e-6 * (1.0 + abs(fi))
        out["ref_obj"] = f
    xref = x if ref_ok else None

    events: list = []
    cnt = {
        "probes": 0,
        "probes_with_g": 0,
        "dbbt_calls": 0,
        "dbbt_cols": 0,
        "oracle_none": 0,
        "ref_checks": 0,
    }

    _orig_sel = O._ns_select
    _orig_dbbt = O.dbbt_on_relaxation
    _orig_dual = O.get_exact_dual_lp_solver

    def dual_w(*a, **k):
        r = _orig_dual(*a, **k)
        if r is None:
            cnt["oracle_none"] += 1  # raw accepted with no g at all on this sweep
        return r

    def sel_w(raw, g, *, require_ns, strict):
        res = _orig_sel(raw, g, require_ns=require_ns, strict=strict)
        if strict:
            return res
        cnt["probes"] += 1
        cnt["probes_with_g"] += g is not None
        if res is None or (g is not None and not res > g):
            return res
        fr = sys._getframe(1)
        if fr.f_code.co_name == "_ns_clamp":  # run_obbt
            fr = fr.f_back
            lo, hi, eps, n_orig = fr.f_locals["lb"], fr.f_locals["ub"], 1e-8, len(lb0)
        else:  # run_obbt_on_relaxation
            lo, hi = fr.f_locals["lb_arr"], fr.f_locals["ub_arr"]
            eps, n_orig = fr.f_locals["eps"], fr.f_locals["n_orig"]
        j = int(fr.f_locals["var_idx"])
        sgn = float(fr.f_locals["c"][j])
        bound = res if sgn > 0 else -res
        safe = None if g is None else (g if sgn > 0 else -g)
        applied = bound > lo[j] + eps if sgn > 0 else bound < hi[j] - eps
        ev = {
            "kind": "a" if g is not None else "b",
            "col": j,
            "orig": j < n_orig,
            "side": "lb" if sgn > 0 else "ub",
            "raw": float(raw),
            "g": None if g is None else float(g),
            "excess": None if g is None else float(raw - g),
            "applied": bool(applied),
            "cur": float(lo[j] if sgn > 0 else hi[j]),
        }
        if xref is not None and j < n_orig:
            cnt["ref_checks"] += 1
            xv = float(xref[j])
            ev["xref"] = xv
            ev["ref_viol"] = (bound - xv) if sgn > 0 else (xv - bound)
            if safe is not None:
                ev["ref_viol_safe"] = (safe - xv) if sgn > 0 else (xv - safe)
        events.append(ev)
        return res

    def dbbt_w(relaxation, n_orig, incumbent_cutoff, **kw):
        res = _orig_dbbt(relaxation, n_orig, incumbent_cutoff, **kw)
        if O._obbt_ns_strict():
            return res
        cnt["dbbt_calls"] += 1
        tok = solver_tuning.set_current(solver_tuning.current().replace(obbt_ns_strict=True))
        try:
            strict = _orig_dbbt(relaxation, n_orig, incumbent_cutoff, **kw)
        finally:
            solver_tuning.reset_current(tok)
        k = min(len(res.tightened_lb), len(strict.tightened_lb))
        cnt["dbbt_cols"] += k
        for j in range(k):
            for side, leg, st in (
                ("lb", res.tightened_lb[j], strict.tightened_lb[j]),
                ("ub", res.tightened_ub[j], strict.tightened_ub[j]),
            ):
                beyond = leg > st if side == "lb" else leg < st
                if not beyond:
                    continue
                ev = {
                    "kind": "dbbt",
                    "col": j,
                    "orig": True,
                    "side": side,
                    "legacy": float(leg),
                    "strict": float(st),
                    "excess": float(abs(leg - st)),
                    "strict_abstained": strict.n_lp_solves == 0,
                }
                if xref is not None:
                    cnt["ref_checks"] += 1
                    xv = float(xref[j])
                    ev["xref"] = xv
                    ev["ref_viol"] = (leg - xv) if side == "lb" else (xv - leg)
                    ev["ref_viol_safe"] = (st - xv) if side == "lb" else (xv - st)
                events.append(ev)
        return res

    def run_root():
        return O.obbt_tighten_root(
            m,
            lb0.copy(),
            ub0.copy(),
            rounds=3,
            incumbent_cutoff=cutoff,
            time_limit_per_lp=1.0,
            deadline=time.perf_counter() + 90.0,
        )

    O._ns_select, O.dbbt_on_relaxation, O.get_exact_dual_lp_solver = sel_w, dbbt_w, dual_w
    try:
        r_off = run_root()
    finally:
        O._ns_select, O.dbbt_on_relaxation, O.get_exact_dual_lp_solver = (
            _orig_sel,
            _orig_dbbt,
            _orig_dual,
        )
    tok = solver_tuning.set_current(solver_tuning.current().replace(obbt_ns_strict=True))
    try:
        r_on = run_root()
    finally:
        solver_tuning.reset_current(tok)

    def box_excl(r):
        if xref is None:
            return None
        lo, hi = np.asarray(r.lb), np.asarray(r.ub)
        return float(max(np.max(lo - xref, initial=0.0), np.max(xref - hi, initial=0.0)))

    out.update(cnt)
    out.update(
        events=events,
        infeasible_off=bool(r_off.infeasible),
        infeasible_on=bool(r_on.infeasible),
        final_excl_off=box_excl(r_off),
        final_excl_on=box_excl(r_on),
        n_tight_off=int(r_off.n_tightened),
        n_tight_on=int(r_on.n_tightened),
    )
    return out


# --------------------------------------------------------------------------- parent
def _names(corpus: str, max_n: int) -> list[str]:
    if corpus == "minlplib":
        d = REPO / "python" / "tests" / "data" / "minlplib_nl"
        return sorted(p.stem for p in d.glob("*.nl"))
    if corpus == "qplib":
        rows = list(csv.DictReader((BENCH / "qplib" / "qplib_manifest.csv").open()))
        # usable_oracle gate (docs/dev/qplib-corpus.md); OBBT needs a finite box to
        # probe, so prefer smaller instances. Selection is by rule, never by name.
        rows = [
            r
            for r in rows
            if r["usable_oracle"] == "True" and int(r["ncons"]) >= 1 and 2 <= int(r["nvars"]) <= 400
        ]
        rows.sort(key=lambda r: (int(r["nvars"]), r["name"]))
        return [r["name"] for r in rows][: max_n * 2]  # some never probe; stop at max_n
    return [
        "pyomo.constrained_layout_clay0203:hull",
        "pyomo.constrained_layout_clay0203:big-m",
        "pyomo.ex_633_trespalacios:hull",
        "pyomo.ex_633_trespalacios:big-m",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["minlplib", "qplib", "gdplib"])
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N candidates")
    ap.add_argument("--out")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--child", nargs=2)
    a = ap.parse_args()
    if a.child:
        print("RESULT " + json.dumps(child(*a.child)), flush=True)
        return 0
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_OBBT_NS_STRICT"}
    n_probed = n_inst = 0
    with open(a.out, "a") as fh:
        for name in _names(a.corpus, a.max)[a.offset :]:
            if a.corpus == "qplib" and n_probed >= a.max:
                break
            t0 = time.time()
            try:
                p = subprocess.run(
                    [sys.executable, "-u", __file__, "--child", a.corpus, name],
                    capture_output=True,
                    text=True,
                    timeout=a.timeout,
                    env=env,
                )
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                rec = (
                    json.loads(lines[-1][7:])
                    if lines
                    else {"corpus": a.corpus, "name": name, "error": p.stderr[-1500:]}
                )
            except subprocess.TimeoutExpired:
                rec = {"corpus": a.corpus, "name": name, "error": "timeout"}
            rec["wall"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_inst += 1
            n_probed += rec.get("probes", 0) > 0
            ev = rec.get("events", [])
            print(
                f"{name}: probes={rec.get('probes')} unsafe={len(ev)} "
                f"ref_ok={rec.get('ref_feasible')} err={str(rec.get('error', ''))[-120:]!r}",
                flush=True,
            )
    print(f"instances={n_inst} with_probes={n_probed}", flush=True)
    return 0


def summarize(paths: list[str]) -> int:
    """Aggregate jsonl files; executed-check count + decision (§6)."""
    recs = [json.loads(ln) for p in paths for ln in Path(p).read_text().splitlines() if ln.strip()]
    tot = {"probes": 0, "probes_with_g": 0, "dbbt_calls": 0, "ref_checks": 0, "oracle_none": 0}
    kinds: dict = {}
    max_ex = 0.0
    excl, excl_unsafe_only, excl_applied, excl_big = [], [], [], []
    for r in recs:
        for k in tot:
            tot[k] += r.get(k, 0)
        for e in r.get("events", []):
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
            # Bounds of magnitude ~1e20 are the LP infinity sentinel; their excess
            # is float spacing, not a tightening, so they are kept out of the max.
            mag = abs(e.get("raw", e.get("legacy", 0.0)))
            if e.get("excess") is not None and mag < 1e15:
                max_ex = max(max_ex, e["excess"])
            v = e.get("ref_viol")
            if v is not None and v > 0:
                excl.append((r["name"], e))
                if v > TOL_EXCL:
                    excl_big.append((r["name"], e))
                if v > TOL_EXCL and e.get("ref_viol_safe", 1.0) <= 0:
                    excl_unsafe_only.append((r["name"], e))
                if v > TOL_EXCL and e.get("applied", True):
                    excl_applied.append((r["name"], e))
    n_ref = sum(1 for r in recs if r.get("ref_feasible"))
    max_viol = max([e["ref_viol"] for _, e in excl], default=0.0)
    print(
        f"instances={len(recs)} errors={sum('error' in r for r in recs)} "
        f"ref_points_verified={n_ref}"
    )
    print(
        "totals", tot, "unsafe events by kind", kinds, f"max excess (|bound| < 1e15)={max_ex:.3g}"
    )
    print(
        f"ref excluded >0: {len(excl)} (max {max_viol:.3g}); "
        f">1e-6: {len(excl_big)}; >1e-6 and applied: {len(excl_applied)}; "
        f">1e-6 with safe bound NOT excluding: {len(excl_unsafe_only)}"
    )
    for nm, e in sorted(excl, key=lambda t: -t[1]["ref_viol"])[:5]:
        print("  ", nm, e)
    fo = [
        (r["name"], r["final_excl_off"], r["final_excl_on"])
        for r in recs
        if r.get("final_excl_off") is not None and r["final_excl_off"] > TOL_EXCL
    ]
    print("final-box excludes ref by >1e-6 (name, OFF, ON):", fo)
    print("executed check count", tot["probes"] + tot["dbbt_calls"] + tot["ref_checks"])
    on = bool(excl_big) or bool(fo)
    print(
        "DECISION (OBBT/DBBT half of the rule; the certified-result half is the OFF/ON "
        "solve panel):",
        "DEFAULT-ON" if on else "keep OFF",
    )
    return 2 if tot["probes"] == 0 else 0


def compact(paths: list[str], out: str) -> None:
    """Per-instance record without the full event list (the committed data form)."""
    with open(out, "w") as fh:
        for p in paths:
            for ln in Path(p).read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                ev = r.pop("events", [])
                r["n_unsafe"] = {k: sum(e["kind"] == k for e in ev) for k in ("a", "b", "dbbt")}
                r["n_unsafe_applied"] = sum(bool(e.get("applied", True)) for e in ev)
                r["max_excess"] = max([e.get("excess") or 0.0 for e in ev], default=None)
                r["max_ref_viol"] = max(
                    [e["ref_viol"] for e in ev if "ref_viol" in e], default=None
                )
                r["top_by_excess"] = sorted(ev, key=lambda e: -(e.get("excess") or 0.0))[:3]
                r["top_by_ref_viol"] = sorted(
                    [e for e in ev if "ref_viol" in e], key=lambda e: -e["ref_viol"]
                )[:3]
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--compact":
        compact(sys.argv[3:], sys.argv[2])
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        sys.exit(summarize(sys.argv[2:]))
    sys.exit(main())
