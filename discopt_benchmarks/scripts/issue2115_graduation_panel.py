#!/usr/bin/env python3
"""Graduation panel for the #2115 / #2142 LP-layer flags (``--flag``):

* ``DISCOPT_NS_PROJECT`` -- exact dual projection as the last rung of the MILP
  driver's node NS bound (default; graduated 2026-11-03);
* ``DISCOPT_SUBTOL_NOISE_GLOBAL`` -- the #1595 sub-``tol`` noise test measured
  against ``‖a_j‖₁·‖y‖∞`` (the scale of ``y``'s error), not only a column's own
  terms;
* ``DISCOPT_LP_DUAL_REBUILD`` (#2142, retired by its panel) -- at the warm
  dual's empty-ratio-test exit, refactorize and tolerate a leaving row within
  ``1e-7(1+|bound|)`` instead of handing the LP to the cold primal. Run it with
  ``--deterministic --max-nodes N --assert-unfired-identical``: an instance on
  which ``DualRebuilds`` is 0 in the ON arm executed the same code as OFF, so
  (when neither arm reached the wall backstop) its status, objective, bound and
  node count must be bit-identical.

The CLAUDE.md §5 differential panel for a bound-changing gate: flag ON vs OFF,
requiring BOTH

  1. *cert-clean* -- no incumbent past its reference optimum, no dual bound
     crossing that optimum or its own incumbent, no certification regression
     (``gap_certified=True`` OFF must not become False ON), no objective drift
     between two arms that both certified, and every incumbent independently
     feasibility-verified against the pristine model (false-primal screen); AND
  2. *net-positive* -- measurably helpful: certified-instance count, dual bound,
     node count / wall, reported per instance and in total.

**Proving the mechanism fired (CLAUDE.md §6).** Each solve runs under
``DISCOPT_PROFILE=1`` bracketed by ``profile_reset_py``/``profile_counters_py``.
For ``DISCOPT_NS_PROJECT``, ``NodeNsProjectRescue`` on the ON arm counts node bounds
the projection produced where every other rung abstained; for
``DISCOPT_SUBTOL_NOISE_GLOBAL``, ``SubtolPivots`` OFF minus ON counts noise-driven
pivots no longer taken. The panel exits 2 (inconclusive) if the mechanism never
fired over the whole panel.

Arms run in separate processes (the flag is a Rust ``OnceLock``, read once per
process), interleaved per instance with the order alternating by index (rule 9).

Corpus: the vendored ``python/tests/data/minlplib_nl`` plus any ``--extra-dir``
(MINLPLib ``.nl`` files fetched from minlplib.org). Oracle: ``--solu`` (MINLPLib
``minlplib.solu``, ``=opt=`` lines) with ``docs/dev/data/cert-optima.json`` as
fallback.

The panel emulates the default install: ``deterministic=False`` (the route a user
gets, and the one on which the m6/m7 class reaches the MILP driver) and
``pyscipopt`` blocked (it is not a dependency; with it importable the #2069
confirmation goes to SCIP and the in-house driver barely runs).

Usage::

    python -u discopt_benchmarks/scripts/issue2115_graduation_panel.py [--flag F] \\
        --time-limit 45 --solu minlplib.solu --extra-dir <dir> --jobs 2

Exit codes: 0 = PASS (cert-clean and the projection fired), 2 = cert-clean but
never fired, 1 = any violation or zero executed assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent.parent
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_OPTIMA = _REPO / "docs" / "dev" / "data" / "cert-optima.json"

_TOL_REL = 1e-4
_TOL_ABS = 1e-6
#: Per-flag panel configuration. ``fired(off, on)`` counts what the flag's mechanism
#: did on one instance (summed for the anti-vacuity check, §6), ``ran(off, on)``
#: says whether the ON arm executed a different code path at all (a bound change
#: on an instance where it did not is time-limit noise), ``leak(off)`` must be 0
#: (the OFF arm honoured the flag).
_FLAGS = {
    "DISCOPT_NS_PROJECT": {
        "fired": lambda off, on: on["NodeNsProjectRescue"],
        "ran": lambda off, on: on["NodeNsProjectTry"] > 0,
        "leak": lambda off: off["NodeNsProjectTry"],
        "show": ("NodeNsAbstain", "NodeNsProjectRescue"),
    },
    # #2115 follow-up: the sub-`tol` noise test measured against `‖a_j‖₁·‖y‖∞`.
    # Its effect is a noise-driven sub-`tol` pivot (and the `Numerical` verdict
    # it ends in) NOT taken, so it fires where the OFF arm's count exceeds ON's.
    "DISCOPT_SUBTOL_NOISE_GLOBAL": {
        "fired": lambda off, on: max(0, off["SubtolPivots"] - on["SubtolPivots"]),
        "ran": lambda off, on: (
            off["SubtolPivots"] != on["SubtolPivots"]
            or off["LpVerdictNumerical"] != on["LpVerdictNumerical"]
        ),
        "leak": lambda off: 0,
        "show": ("SubtolPivots", "LpVerdictNumerical"),
    },
    # #2142 B: RETIRED by this panel (docs/dev/data/2142-flag-panel/); the build no
    # longer has the DualRebuild* counters, so a fresh run reports them missing.
    # Kept so ``--rows`` re-derives the recorded verdict.
    # The rebuild path is entered only ON (``tolerated`` is empty OFF), so
    # any ``DualRebuilds`` OFF is a leak and ``DualRebuilds`` ON is where it ran.
    "DISCOPT_LP_DUAL_REBUILD": {
        "fired": lambda off, on: on["DualRebuilds"],
        "ran": lambda off, on: on["DualRebuilds"] > 0,
        "leak": lambda off: off["DualRebuilds"],
        "show": ("DualRebuilds", "DualRebuildOptimal"),
    },
    # #2142 item 3: SCIP's branching-point rule in the native spatial kernel
    # (``spatial_clamp`` / ``spatial_midpull_reldomtrig``). RETIRED by its panel
    # (docs/dev/data/2142-branching/, performance-plan §107); the build no longer
    # has the SpatialSplit* counters. Kept so ``--rows`` re-derives the verdict.
    # Only a spatial split whose point the rule moved can change the search, so
    # ``SpatialSplitMoved`` ON is where it ran; OFF it must be 0.
    "DISCOPT_SPATIAL_BRANCH_CLAMP": {
        "fired": lambda off, on: on["SpatialSplitMoved"],
        "ran": lambda off, on: on["SpatialSplitMoved"] > 0,
        "leak": lambda off: off["SpatialSplitMoved"],
        "show": ("SpatialSplits", "SpatialSplitMoved"),
    },
}
_CTRS = (
    "NodeNsProjectTry",
    "NodeNsProjectRescue",
    "NodeNsProjectSingular",
    "NodeNsAbstain",
    "SubtolPivots",
    "LpVerdictNumerical",
    "WarmSubtolRepairFailed",
    "UnboundedRejectObjective",
    "DualRebuilds",
    "DualRebuildCleared",
    "DualRebuildTolerated",
    "DualRebuildBails",
    "DualRebuildOptimal",
    "DualRebuildAuditFail",
    "SpatialSplits",
    "SpatialSplitMoved",
)
_FLAG = "DISCOPT_NS_PROJECT"  # set from --flag in main()

_RUNNER = """
import json, sys, time

if sys.argv[4] == "1":
    sys.modules["pyscipopt"] = None  # emulate the default install (no SCIP)
import numpy as np
import discopt
import discopt.modeling as dm
from discopt._rust import profile_counters_py, profile_reset_py
from discopt.modeling.core import ObjectiveSense

path, tl, det = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "1"
max_nodes = int(sys.argv[5])
kw = {"max_nodes": max_nodes} if max_nodes > 0 else {}
model = dm.from_nl(path)
profile_reset_py()
t0 = time.perf_counter()
res = model.solve(time_limit=tl, deterministic=det, **kw)
wall = time.perf_counter() - t0
ctrs = dict(profile_counters_py())

verified = None
if res.x:
    from discopt._relax.primal_heuristics import passes_false_primal_screen
    from discopt._tape_nlp_evaluator import build_evaluator

    def _jax_evaluator():
        from discopt._relax.nlp_evaluator import cached_evaluator
        return cached_evaluator(model)

    ev = build_evaluator(model, _jax_evaluator)
    flat = []
    for v in model._variables:
        flat.extend(np.asarray(res.x[v.name], dtype=float).ravel().tolist())
    verified = bool(passes_false_primal_screen(ev, np.asarray(flat, dtype=float)))

minimize = model._objective is None or model._objective.sense == ObjectiveSense.MINIMIZE
out = {
    "module": discopt.__file__,
    "status": str(res.status),
    "objective": res.objective,
    "bound": res.bound,
    "gap_certified": bool(res.gap_certified),
    "nodes": int(res.node_count or 0),
    "wall": wall,
    "verified": verified,
    "minimize": bool(minimize),
    "undecided": (res.solver_stats or {}).get("tree/undecided_nodes"),
}
# CLAUDE.md section 8: a build without a counter reads as "never fired"; record it.
out["missing_counters"] = [k for k in CTRS if k not in ctrs]
for k in CTRS:
    out[k] = int(ctrs.get(k, 0))
print("PANELJSON " + json.dumps(out), flush=True)
""".replace("CTRS", repr(_CTRS))


def _solve_once(
    path: Path,
    flag: str,
    time_limit: float,
    deterministic: bool,
    block_scip: bool,
    max_nodes: int = 0,
) -> dict:
    """One solve with ``<--flag>=flag`` in its own process. Never swallows
    a failure (rule 7): a missing PANELJSON line raises. ``PYTHONHASHSEED`` is
    pinned so two processes with the same flag run the same search."""
    env = {**os.environ, _FLAG: flag, "DISCOPT_PROFILE": "1", "PYTHONHASHSEED": "0"}
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            "-c",
            _RUNNER,
            str(path),
            str(time_limit),
            "1" if deterministic else "0",
            "1" if block_scip else "0",
            str(max_nodes),
        ],
        capture_output=True,
        text=True,
        timeout=time_limit * 8 + 300,
        env=env,
    )
    for ln in proc.stdout.splitlines():
        if ln.startswith("PANELJSON "):
            return json.loads(ln[len("PANELJSON ") :])
    raise RuntimeError(
        f"{path.stem} [{_FLAG}={flag}] produced no result "
        f"(rc={proc.returncode}): {proc.stderr.strip()[-600:]}"
    )


def _load_optima(solu: str | None) -> dict[str, float]:
    optima: dict[str, float] = {}
    if _OPTIMA.exists():
        optima.update(json.loads(_OPTIMA.read_text()))
    if solu:
        for ln in Path(solu).read_text().splitlines():
            parts = ln.split()
            if len(parts) == 3 and parts[0] == "=opt=":
                optima[parts[1]] = float(parts[2])
    return optima


def _load_best(solu: str | None) -> dict[str, float]:
    """MINLPLib ``=best=`` primal values: unproven optima, but feasible objective
    values, so a valid dual bound can never pass one (a bound check, never an
    incumbent check)."""
    best: dict[str, float] = {}
    if solu:
        for ln in Path(solu).read_text().splitlines():
            parts = ln.split()
            if len(parts) == 3 and parts[0] == "=best=":
                best[parts[1]] = float(parts[2])
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", choices=sorted(_FLAGS), default="DISCOPT_NS_PROJECT")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--solu", default=None)
    ap.add_argument("--extra-dir", default=None)
    ap.add_argument("--instances", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument(
        "--allow-scip",
        action="store_true",
        help="let the solve import pyscipopt. Off by default: SCIP is not a dependency, "
        "and with it present the #2069 confirmation goes to SCIP and the in-house "
        "MILP driver -- where this flag acts -- barely runs (measured: m6/m7 certify "
        "in both arms, 3/3 replicates).",
    )
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="solve with deterministic=True. Off by default: on the m6/m7 class that "
        "route never reaches the MILP driver (measured: m6 spends 30 s in 861 spatial "
        "nodes), so it cannot exercise the flag.",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=0,
        help="node budget per solve (0 = none); with --deterministic it makes a solve a "
        "function of the model, so --time-limit is only a backstop",
    )
    ap.add_argument(
        "--assert-unfired-identical",
        action="store_true",
        help="require status/objective/bound/nodes bit-identical OFF vs ON on every "
        "instance where the mechanism did not run and neither arm reached the wall "
        "backstop (meaningful only with --deterministic and --max-nodes)",
    )
    ap.add_argument(
        "--rows",
        default="",
        help="comma-separated .partial.jsonl files: report from these rows, solve nothing",
    )
    ap.add_argument("--out", default=str(_REPO / "reports" / "issue2115_panel.json"))
    args = ap.parse_args()
    global _FLAG
    _FLAG = args.flag
    cfg = _FLAGS[args.flag]

    optima = _load_optima(args.solu)
    best = _load_best(args.solu)
    paths = sorted(_NL_DIR.glob("*.nl"))
    if args.extra_dir:
        paths += sorted(Path(args.extra_dir).glob("*.nl"))
    if args.instances:
        keep = {n.strip() for n in args.instances.split(",") if n.strip()}
        paths = [p for p in paths if p.stem in keep]

    def run(ix_path):
        i, path = ix_path
        order = ("0", "1") if i % 2 == 0 else ("1", "0")
        arms = {
            f: _solve_once(
                path,
                f,
                args.time_limit,
                args.deterministic,
                not args.allow_scip,
                args.max_nodes,
            )
            for f in order
        }
        off, on = arms["0"], arms["1"]
        row = {
            "instance": path.stem,
            "off": off,
            "on": on,
            "load1": os.getloadavg()[0],
            "time_limit": args.time_limit,
        }
        with open(str(args.out) + ".partial.jsonl", "a") as fh:  # survives a crash
            fh.write(json.dumps(row) + "\n")
        a, b = cfg["show"]
        print(
            f"{path.stem:22s} OFF {off['status']:22s} b={off['bound']} n={off['nodes']} "
            f"{a}={off[a]} {b}={off[b]} | ON {on['status']:22s} b={on['bound']} "
            f"n={on['nodes']} {a}={on[a]} {b}={on[b]} fired={cfg['fired'](off, on)} "
            f"undecided {off.get('undecided')}/{on.get('undecided')} "
            f"wall {off['wall']:.1f}/{on['wall']:.1f} load {row['load1']:.2f}",
            flush=True,
        )
        return row

    if args.rows:
        # Re-derive both bars from rows already written (``.partial.jsonl``), e.g.
        # a panel run in segments; the last row per instance wins.
        by_name: dict[str, dict] = {}
        for fn in args.rows.split(","):
            for ln in Path(fn).read_text().splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    by_name[r["instance"]] = r
        rows = [by_name[n] for n in sorted(by_name)]
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            rows = list(ex.map(run, enumerate(paths)))

    checks = 0
    violations: list[str] = []
    for r in rows:
        name, off, on = r["instance"], r["off"], r["on"]
        ref = optima.get(name)
        for tag, a in (("off", off), ("on", on)):
            checks += 1
            if "/home/user/discopt/" not in a["module"] and str(_REPO) not in a["module"]:
                violations.append(f"{name}[{tag}]: loaded discopt from {a['module']}")
            # Only the counters this flag's probe reads must exist: a retired flag's
            # counters (``DualRebuild*``) are absent from later builds by design.
            missing = [k for k in a.get("missing_counters") or [] if k in cfg["show"]]
            if missing:
                violations.append(f"{name}[{tag}]: build lacks {missing}")
            if a["verified"] is False:
                violations.append(f"{name}[{tag}]: incumbent failed the false-primal screen")
            if a["verified"] is not None:
                checks += 1
            sgn = 1.0 if a["minimize"] else -1.0
            if a["objective"] is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(a["objective"])) + _TOL_ABS
                if sgn * (a["bound"] - a["objective"]) > slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} past incumbent")
            if ref is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(ref)) + _TOL_ABS
                if sgn * (a["bound"] - ref) > slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} past optimum {ref}")
            primal = best.get(name)
            if ref is None and primal is not None and a["bound"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(primal)) + _TOL_ABS
                if sgn * (a["bound"] - primal) > slack:
                    violations.append(f"{name}[{tag}]: bound {a['bound']} past =best= {primal}")
            if ref is not None and a["objective"] is not None:
                checks += 1
                slack = _TOL_REL * max(1.0, abs(ref)) + _TOL_ABS
                if sgn * (ref - a["objective"]) > slack:
                    violations.append(f"{name}[{tag}]: incumbent {a['objective']} past {ref}")
        checks += 1
        if off["gap_certified"] and not on["gap_certified"]:
            violations.append(f"{name}: certification lost with the flag ON")
        if off["gap_certified"] and on["gap_certified"]:
            checks += 1
            o0, o1 = off["objective"], on["objective"]
            if abs(o1 - o0) > _TOL_REL * max(1.0, abs(o0)) + _TOL_ABS:
                violations.append(f"{name}: certified objectives differ {o0} vs {o1}")

    # Bound-neutral where the mechanism never ran (CLAUDE.md section 5): the ON arm
    # executed the OFF arm's code, so a deterministic, work-terminated solve must
    # reproduce it exactly. A row where either arm reached the wall backstop is
    # machine-speed dependent and is counted apart, never silently passed.
    identical_checked, identical_skipped_wall = 0, []
    if args.assert_unfired_identical:
        for r in rows:
            name, off, on = r["instance"], r["off"], r["on"]
            if cfg["ran"](off, on):
                continue
            if max(off["wall"], on["wall"]) >= 0.95 * r.get("time_limit", args.time_limit):
                identical_skipped_wall.append(name)
                continue
            identical_checked += 1
            checks += 1
            for k in ("status", "objective", "bound", "nodes"):
                if off[k] != on[k]:
                    violations.append(
                        f"{name}: mechanism did not run but {k} differs "
                        f"(OFF {off[k]!r} vs ON {on[k]!r})"
                    )
    rescued = sum(cfg["fired"](r["off"], r["on"]) for r in rows)
    off_leak = sum(cfg["leak"](r["off"]) for r in rows)
    if off_leak:
        violations.append(f"OFF arm ran the mechanism {off_leak} times (flag not honoured)")
    fired_on = [r["instance"] for r in rows if cfg["fired"](r["off"], r["on"]) > 0]
    cert_off = sum(r["off"]["gap_certified"] for r in rows)
    cert_on = sum(r["on"]["gap_certified"] for r in rows)
    gained = [
        r["instance"] for r in rows if r["on"]["gap_certified"] and not r["off"]["gap_certified"]
    ]
    tighter, looser = [], []
    for r in rows:
        bo, bn = r["off"]["bound"], r["on"]["bound"]
        if bo is None or bn is None:
            continue
        sgn = 1.0 if r["off"]["minimize"] else -1.0
        tol = _TOL_REL * max(1.0, abs(bo)) + _TOL_ABS
        if sgn * (bn - bo) > tol:
            tighter.append(r["instance"])
        elif sgn * (bo - bn) > tol:
            looser.append(r["instance"])
    diff_rows = [r for r in rows if cfg["ran"](r["off"], r["on"])]
    # A bound/certification difference on an instance where the ON arm never ran a
    # different code path cannot be the flag's doing (time-limit noise); reported
    # apart.
    ran = {r["instance"] for r in diff_rows}
    tighter_ran = [i for i in tighter if i in ran]
    looser_ran = [i for i in looser if i in ran]
    nodes_off = sum(r["off"]["nodes"] for r in rows)
    nodes_on = sum(r["on"]["nodes"] for r in rows)
    wall_off = sum(r["off"]["wall"] for r in rows)
    wall_on = sum(r["on"]["wall"] for r in rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "flag": args.flag,
                "time_limit": args.time_limit,
                "deterministic": args.deterministic,
                "max_nodes": args.max_nodes,
                "load_at_start": os.getloadavg(),
                "unfired_identical_checked": identical_checked,
                "unfired_identical_skipped_wall": identical_skipped_wall,
                "scip_blocked": not args.allow_scip,
                "instances": len(rows),
                "executed_checks": checks,
                "violations": violations,
                "rescued": rescued,
                "fired_on": fired_on,
                "certified_off": cert_off,
                "certified_on": cert_on,
                "certification_gained": gained,
                "bounds_tighter": tighter,
                "bounds_looser": looser,
                "bounds_tighter_where_ran": tighter_ran,
                "bounds_looser_where_ran": looser_ran,
                "nodes_off": nodes_off,
                "nodes_on": nodes_on,
                "wall_off": wall_off,
                "wall_on": wall_on,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\ninstances           : {len(rows)}")
    print(f"executed checks     : {checks}")
    print(f"violations          : {len(violations)}")
    for v in violations:
        print("  !", v)
    print(f"mechanism fired     : {rescued} on {len(fired_on)} instance(s) {fired_on}")
    if args.assert_unfired_identical:
        print(
            f"unfired identical   : {identical_checked} checked, "
            f"{len(identical_skipped_wall)} skipped (wall backstop) {identical_skipped_wall}"
        )
    print(f"certified           : OFF {cert_off} -> ON {cert_on}  gained {gained}")
    print(f"bounds tighter      : {len(tighter)} {tighter}  (where it ran: {tighter_ran})")
    print(f"bounds looser       : {len(looser)} {looser}  (where it ran: {looser_ran})")
    print(f"nodes (all)         : OFF {nodes_off} -> ON {nodes_on}")
    print(f"wall  (all)         : OFF {wall_off:.1f}s -> ON {wall_on:.1f}s")
    if diff_rows:
        w_off = sum(r["off"]["wall"] for r in diff_rows)
        w_on = sum(r["on"]["wall"] for r in diff_rows)
        print(
            f"wall  (fired)       : OFF {w_off:.1f}s -> ON {w_on:.1f}s "
            f"over {len(diff_rows)} instance(s)"
        )
    print(f"report              : {out}")
    if checks == 0:
        print("\nVERDICT: FAIL - zero executed checks.")
        return 1
    if violations:
        print("\nVERDICT: FAIL - cert-clean violated.")
        return 1
    if rescued == 0:
        print("\nVERDICT: INCONCLUSIVE - the mechanism never fired.")
        return 2
    print("\nVERDICT: cert-clean, mechanism fired; read the benefit columns above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
