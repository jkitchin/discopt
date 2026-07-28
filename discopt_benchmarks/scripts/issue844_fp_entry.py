"""#844 entry experiment for the LP-based feasibility pump — BEFORE any implementation.

SCIP's own attribution on our gap instances names the constructor: ``gastrans040``'s
incumbent comes from ``feaspump`` at node 1, ``gastrans582_cold13``'s from
``pscostdiving`` at node 12 (``docs/dev/sota-parity-analysis-2026-07-27.md`` §G-G).
This script answers the three questions that must be settled with a measurement on
**real corpus instances** before writing a pump (CLAUDE.md §4, and the #727 RLT lesson:
a mechanism validated on a synthetic proxy can be a no-op on the real class).

Q1  What primal constructors does discopt have TODAY? Answered by reading source, not
    by this script; the script records the one fact that decides reachability — the
    scope gate of the engine that already hosts an LP-based objective pump
    (``_jax/lp_spatial_bb.py``'s ``feasibility_pump``, Fischetti-Glover-Lodi).

Q2  Is there a fractional LP solution to pump at all? A pump alternates between the
    relaxation and a rounding; if the root McCormick LP never solves, the pump has
    nothing to read and the gap is upstream of any primal work. Stage ``root``
    reproduces the engine's own root sequence — ``classify_nonlinear_terms`` →
    ``flat_variable_bounds`` → root OBBT when the box has an infinite endpoint →
    ``_relax_bound`` — and reports whether a point came back and how many INTEGER
    coordinates of it are fractional.

Q3  What is today's primal gap and time-to-first-incumbent, so that "better" is a
    number? Stage ``solve`` runs the DEFAULT path at two budgets and scores the
    incumbent against ``minlplib.solu``.

    Time-to-first-incumbent is bracketed by two budgets rather than instrumented with
    an ``incumbent_callback``: passing one changes which engine runs. Read from source
    this run — ``solver.py:570`` declines the native spatial kernel outright when
    ``incumbent_callback is not None``. A callback-instrumented timing would therefore
    be a measurement of a different code path than the default, which is the class of
    error CLAUDE.md §8 is about.

Corpus is resolved through ``discopt_benchmarks/utils/corpus.py`` (never a hardcoded
Dropbox path). The panel is the vendored 50-instance BARON head-to-head list — the set
the "40 jointly-proved slow-ratio" figure is computed over — plus the three §G-G
targets.

Executed-assertion discipline (CLAUDE.md §6): every probe increments a counter and the
script exits non-zero when the count is zero, so a harness that traversed nothing can
never read as a pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "discopt_benchmarks"))

from utils.corpus import nl_dir, solu_path, warn_if_synced  # noqa: E402

TARGETS = ["gastrans040", "gastrans582_cold13", "watercontamination0202"]
PANEL_LIST = _REPO / "discopt_benchmarks" / "config" / "baron_global50.txt"

# Root probe: the engine budgets its own root OBBT at a third of the remaining time
# (lp_spatial_bb.py:501). Mirror that shape rather than inventing a schedule.
ROOT_BUDGET_S = 120.0
ROOT_CHILD_TIMEOUT_S = 400.0

CHILD_ROOT = r'''
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import discopt

# CLAUDE.md section 8: prove WHICH code is loaded before measuring anything with it.
repo = sys.argv[3]
assert discopt.__file__.startswith(repo), discopt.__file__
from discopt._jax import lp_spatial_bb as LSB
assert LSB.__file__.startswith(repo), LSB.__file__
# Positive marker unique to the version under test: the LP objective pump this
# experiment is about is a closure inside the engine, so assert its docstring text.
_src = open(LSB.__file__).read()
assert "Objective feasibility pump (Fischetti-Glover-Lodi)" in _src, "pump marker absent"
assert hasattr(LSB, "_is_in_scope"), "marker absent: _is_in_scope"
assert hasattr(LSB, "_relax_bound"), "marker absent: _relax_bound"

import discopt.modeling as dm
from discopt.modeling.core import ObjectiveSense, VarType
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.term_classifier import classify_nonlinear_terms

inst, budget = sys.argv[1], float(sys.argv[2])
out = {"instance": inst, "asserts": 0}
t0 = time.perf_counter()
m = dm.from_nl(inst)
out["t_load"] = time.perf_counter() - t0

# SENSE READ PER MODEL. The .nl reader does not normalise to MINIMIZE.
obj = m._objective
out["sense"] = "min" if (obj is None or obj.sense == ObjectiveSense.MINIMIZE) else "max"
n_int = sum(int(v.size) for v in m._variables
            if v.var_type in (VarType.INTEGER, VarType.BINARY))
n_all = sum(int(v.size) for v in m._variables)
out["n_vars"] = n_all
out["n_int"] = n_int
out["n_cont"] = n_all - n_int
out["n_cons"] = len(m._constraints)

# Q1 reachability: does the engine that HOSTS the LP pump accept this model?
out["in_scope_pure"] = bool(LSB._is_in_scope(m, mixed=False))
out["in_scope_mixed"] = bool(LSB._is_in_scope(m, mixed=True))
out["asserts"] += 2

# Q2: the root McCormick LP the pump would read.
lb, ub = flat_variable_bounds(m)
lb = np.asarray(lb, dtype=float).copy()
ub = np.asarray(ub, dtype=float).copy()
is_int = LSB._integer_mask(m)
out["inf_box"] = not (bool(np.all(np.isfinite(lb))) and bool(np.all(np.isfinite(ub))))

t0 = time.perf_counter()
terms = classify_nonlinear_terms(m)
out["t_classify"] = time.perf_counter() - t0

def probe(lo, hi, tag):
    """One root-LP attempt. Returns the (bound, x) or None; records timing per tag."""
    s = time.perf_counter()
    try:
        r = LSB._relax_bound(m, terms, lo, hi, deadline=time.perf_counter() + budget)
    except BaseException as exc:
        # Do NOT swallow (section 7): record the type and re-raise shape into the record.
        out["root_error_" + tag] = type(exc).__name__ + ": " + str(exc)[:300]
        out["t_relax_" + tag] = time.perf_counter() - s
        return None
    out["t_relax_" + tag] = time.perf_counter() - s
    return r

r = probe(lb, ub, "raw")
out["root_lp_raw_ok"] = r is not None
used = "raw"

if r is None and out["inf_box"]:
    # The engine runs root OBBT when the box has an infinite endpoint
    # (lp_spatial_bb.py:492) precisely so the relaxation gets a valid bound.
    s = time.perf_counter()
    try:
        from discopt._jax.obbt import obbt_tighten_root
        rr = obbt_tighten_root(m, lb, ub, rounds=5,
                               deadline=time.perf_counter() + budget / 3.0,
                               time_limit_per_lp=0.5)
        out["obbt_infeasible"] = bool(rr.infeasible)
        if not rr.infeasible:
            _rlb = np.asarray(rr.lb, dtype=float); _rub = np.asarray(rr.ub, dtype=float)
            lb = np.maximum(lb, np.where(is_int, np.floor(_rlb + 1e-9), _rlb))
            ub = np.minimum(ub, np.where(is_int, np.ceil(_rub - 1e-9), _rub))
    except BaseException as exc:
        out["obbt_error"] = type(exc).__name__ + ": " + str(exc)[:300]
    out["t_obbt"] = time.perf_counter() - s
    out["inf_box_after_obbt"] = not (bool(np.all(np.isfinite(lb)))
                                     and bool(np.all(np.isfinite(ub))))
    r = probe(lb, ub, "obbt")
    used = "obbt"

out["root_lp_ok"] = r is not None
out["root_lp_path"] = used if r is not None else None
out["asserts"] += 1

if r is not None:
    bound, x, _info = r
    out["root_bound"] = None if bound is None else float(bound)
    x = np.asarray(x, dtype=float)
    n = int(lb.size)
    xi = x[:n][is_int]
    if xi.size:
        frac = np.abs(xi - np.round(xi))
        out["n_frac_int"] = int(np.count_nonzero(frac > 1e-6))
        out["frac_int_fraction"] = float(np.count_nonzero(frac > 1e-6) / xi.size)
        out["max_frac"] = float(frac.max())
        out["asserts"] += 1
    else:
        out["n_frac_int"] = 0

print("JSONRESULT " + json.dumps(out))
'''

CHILD_SOLVE = r"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import warnings; warnings.filterwarnings("ignore")

import discopt
repo = sys.argv[3]
assert discopt.__file__.startswith(repo), discopt.__file__
import discopt.modeling as dm
from discopt.modeling.core import ObjectiveSense

inst, tl = sys.argv[1], float(sys.argv[2])
m = dm.from_nl(inst)
obj = m._objective
sense = "min" if (obj is None or obj.sense == ObjectiveSense.MINIMIZE) else "max"
t0 = time.perf_counter()
r = m.solve(time_limit=tl)
wall = time.perf_counter() - t0
print("JSONRESULT " + json.dumps({
    "instance": inst, "sense": sense, "time_limit": tl, "wall": wall,
    "status": str(r.status),
    "objective": None if r.objective is None else float(r.objective),
    "bound": None if r.bound is None else float(r.bound),
    "gap_certified": bool(getattr(r, "gap_certified", False)),
}))
"""


CHILD_ENGINE = r"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import warnings; warnings.filterwarnings("ignore")

import discopt
repo = sys.argv[3]
assert discopt.__file__.startswith(repo), discopt.__file__
import discopt.modeling as dm
from discopt.modeling.core import ObjectiveSense
from discopt._jax import lp_spatial_bb as LSB
_src = open(LSB.__file__).read()
assert "Objective feasibility pump (Fischetti-Glover-Lodi)" in _src, "pump marker absent"

inst, tl = sys.argv[1], float(sys.argv[2])
m = dm.from_nl(inst)
obj = m._objective
sense = "min" if (obj is None or obj.sense == ObjectiveSense.MINIMIZE) else "max"

# Exactly the invocation the #844 fallback makes (modeling/core.py:4320-4326), except
# mixed=True -- the one condition that currently declines these models by default.
t0 = time.perf_counter()
r = LSB.solve_lp_spatial_bb(m, time_limit=tl, use_obbt=False,
                            require_incremental=True, mixed=True)
wall = time.perf_counter() - t0
out = {"instance": inst, "sense": sense, "time_limit": tl, "wall": wall}
if r is None:
    # require_incremental declined: the incremental McCormick structure did not build,
    # which is ALSO the condition under which the pump itself is skipped (line 815).
    out["engine_declined"] = True
else:
    out["engine_declined"] = False
    out["status"] = str(r.status)
    out["objective"] = None if r.objective is None else float(r.objective)
    out["bound"] = None if r.bound is None else float(r.bound)
    out["node_count"] = int(r.node_count)
print("JSONRESULT " + json.dumps(out))
"""


def read_oracle() -> dict[str, tuple[str, float | None]]:
    """``{name: (marker, value)}`` parsed from ``minlplib.solu`` IN THIS RUN."""
    p = solu_path()
    if p is None:
        raise SystemExit("FATAL: no minlplib.solu under the resolved corpus root")
    table: dict[str, tuple[str, float | None]] = {}
    with open(p) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            marker, name = parts[0], parts[1]
            val = float(parts[2]) if len(parts) > 2 else None
            table[name] = (marker, val)
    return table


def panel() -> list[str]:
    names = [ln.strip() for ln in PANEL_LIST.read_text().splitlines() if ln.strip()]
    for t in TARGETS:
        if t not in names:
            names.append(t)
    return names


def run_child(src: str, inst_path: Path, arg: float, timeout: float) -> dict:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", "-c", src, str(inst_path), str(arg), str(_REPO)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("JSONRESULT "):
            return json.loads(line[len("JSONRESULT ") :])
    return {
        "child_failed": True,
        "returncode": proc.returncode,
        "stderr": proc.stderr[-800:],
        "wall": time.perf_counter() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("root", "solve", "engine"), required=True)
    ap.add_argument("--budget", type=float, default=ROOT_BUDGET_S)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated instance subset")
    args = ap.parse_args()

    warn_if_synced("issue844_fp_entry")
    nld = nl_dir()
    if nld is None:
        raise SystemExit("FATAL: no corpus (utils.corpus.nl_dir() is None)")

    names = panel()
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        names = [n for n in names if n in want]

    oracle = read_oracle()
    rows: list[dict] = []
    executed = 0
    missing: list[str] = []

    for i, name in enumerate(names, 1):
        p = nld / f"{name}.nl"
        if not p.is_file():
            missing.append(name)
            print(f"[{i}/{len(names)}] {name}: NO .nl IN CORPUS", flush=True)
            continue
        src = {"root": CHILD_ROOT, "solve": CHILD_SOLVE, "engine": CHILD_ENGINE}[args.stage]
        tmo = ROOT_CHILD_TIMEOUT_S if args.stage == "root" else args.budget * 3 + 240
        try:
            rec = run_child(src, p, args.budget, tmo)
        except subprocess.TimeoutExpired:
            rec = {"child_timeout": True}
        rec["instance"] = name
        mk, val = oracle.get(name, ("", None))
        rec["oracle_marker"] = mk
        rec["oracle"] = val
        executed += int(rec.get("asserts", 0)) or (0 if rec.get("child_failed") else 1)
        rows.append(rec)
        if args.stage == "root":
            print(
                f"[{i}/{len(names)}] {name}: vars={rec.get('n_vars')} int={rec.get('n_int')} "
                f"sense={rec.get('sense')} scope(pure/mixed)="
                f"{rec.get('in_scope_pure')}/{rec.get('in_scope_mixed')} "
                f"rootLP={rec.get('root_lp_ok')} via={rec.get('root_lp_path')} "
                f"frac={rec.get('n_frac_int')}/{rec.get('n_int')} "
                f"t_relax={rec.get('t_relax_raw')}",
                flush=True,
            )
        elif args.stage == "engine":
            print(
                f"[{i}/{len(names)}] {name}: declined={rec.get('engine_declined')} "
                f"status={rec.get('status')} obj={rec.get('objective')} "
                f"bound={rec.get('bound')} nodes={rec.get('node_count')} "
                f"wall={rec.get('wall')} oracle={val}",
                flush=True,
            )
        else:
            print(
                f"[{i}/{len(names)}] {name}: status={rec.get('status')} "
                f"obj={rec.get('objective')} bound={rec.get('bound')} "
                f"wall={rec.get('wall')} oracle={val}",
                flush=True,
            )

    Path(args.out).write_text(
        json.dumps({"stage": args.stage, "budget": args.budget, "rows": rows}, indent=1)
    )

    print(f"\nINSTANCES EXAMINED: {len(rows)}  (missing .nl: {len(missing)})")
    print(f"EXECUTED PROBES: {executed}")
    if executed == 0:
        print("FATAL: zero probes executed — this harness measured nothing.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
