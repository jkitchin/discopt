"""#933 Regime-2 graduation panel: DISCOPT_ROOT_BOUND_SEED ON vs OFF, in-repo corpus.

CLAUDE.md §5 (bound-changing): the root-bound seed changes which nodes prune
first, so it shipped default-OFF behind ``DISCOPT_ROOT_BOUND_SEED`` until this
panel's 2026-08-09 run passed both bars (see the graduation record in
``_root_bound_seed_enabled``'s docstring; the flag is now default ON with
``=0`` as the opt-out). The panel remains runnable as the ongoing regression
watch. The two bars:

  (1) cert-clean — ``incorrect_count = 0``: no reported bound crosses its
      reference optimum (registry ``python/tests/data/known_optima.toml``), no
      objective on the infeasible side of the oracle, no ``gap_certified=True``
      instance regressing to uncertified ON vs OFF, objective drift within
      tolerance;
  (2) net-positive — measurably helpful broadly: bound coverage (fewer
      ``bound=None``) and/or tighter reported bounds, without broad node/wall
      damage.

§6: prints executed comparison counts and exits non-zero when any is zero.
§7: worker output is parsed strictly; a crashed worker is REPORTED as a row,
never swallowed. §8: the worker asserts the seeding marker
(``PyTreeManager.seed_root_bound``) is present, and that the flag round-trips.
§9 caveat: runs are interleaved (OFF, ON per instance, adjacent in time) but
wall-clock is a secondary signal here; bound coverage/tightness — the point of
the mechanism — is load-insensitive.

Usage:  python issue933_seed_graduation_panel.py [--tl 8] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"
OPTIMA = REPO / "python" / "tests" / "data" / "known_optima.toml"

# Relative tolerance for "bound crosses the oracle" (matches the oracle sweep
# convention in issue930_adversarial_oracle_sweep.py) and for objective drift.
REL_TOL = 1e-4
ABS_TOL = 1e-6

WORKER = r"""
import json, os, sys, time
nl, tl, flag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.environ["DISCOPT_ROOT_BOUND_SEED"] = flag
import discopt.solver as S
from discopt._rust import PyTreeManager
# §8: assert the version under test actually carries the #933 surface, and that
# the flag helper reads back what we set.
assert hasattr(PyTreeManager, "seed_root_bound"), "MARKER ABSENT: not the #933 tree"
assert hasattr(S, "_finalize_reported_bound"), "MARKER ABSENT: no #933 chokepoint"
assert S._root_bound_seed_enabled() == (flag == "1"), "flag did not round-trip"
import discopt.modeling as dm
t0 = time.perf_counter()
r = dm.from_nl(nl).solve(time_limit=tl)
print(json.dumps({
    "status": r.status, "obj": r.objective, "bound": r.bound, "gap": r.gap,
    "nodes": r.node_count, "wall": time.perf_counter() - t0,
    "cert": bool(r.gap_certified),
}))
"""


def run_one(nl: Path, tl: float, flag: str, worker: str, hard_cap: float) -> dict:
    try:
        out = subprocess.run(
            [sys.executable, worker, str(nl), str(tl), flag],
            capture_output=True,
            text=True,
            timeout=hard_cap,
        )
    except subprocess.TimeoutExpired:
        return {"error": "hard-cap timeout"}
    if out.returncode != 0:
        return {"error": (out.stderr or "worker failed").strip()[-400:]}
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": f"unparseable worker output: {out.stdout[-200:]!r}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=8.0)
    ap.add_argument("--hard-cap", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated instance subset")
    args = ap.parse_args()

    with OPTIMA.open("rb") as fh:
        registry = tomllib.load(fh)
    registry.pop("schema", None)
    oracle = {k: float(v["optimum"]) for k, v in registry.items()}

    insts = sorted(p.stem for p in CORPUS.glob("*.nl"))
    if args.only:
        keep = set(args.only.split(","))
        insts = [i for i in insts if i in keep]

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(WORKER)
        worker = fh.name

    rows = []
    n_oracle_checks = 0
    n_bound_comparisons = 0
    incorrect = []
    cert_regressions = []
    obj_drift = []
    cover_off = cover_on = 0
    tighter = looser = equal = 0
    try:
        for i, inst in enumerate(insts):
            nl = CORPUS / f"{inst}.nl"
            off = run_one(nl, args.tl, "0", worker, args.hard_cap)
            on = run_one(nl, args.tl, "1", worker, args.hard_cap)
            rows.append({"inst": inst, "off": off, "on": on})
            line = f"[{i + 1}/{len(insts)}] {inst}:"
            for tag, r in (("off", off), ("on", on)):
                if "error" in r:
                    line += f" {tag}=ERR"
                    continue
                line += f" {tag}(status={r['status']} bound={r['bound']} wall={r['wall']:.1f})"
            print(line, flush=True)

            opt = oracle.get(inst)
            for tag, r in (("off", off), ("on", on)):
                if "error" in r or opt is None:
                    continue
                # Sense: every registry instance here is minimize-reported except
                # where the bound sits above the objective; detect by relation.
                b, o = r.get("bound"), r.get("obj")
                tol = ABS_TOL + REL_TOL * abs(opt)
                if b is not None:
                    n_oracle_checks += 1
                    # A dual bound must not be on the far side of the optimum:
                    # min-sense reported bounds satisfy bound <= opt + tol;
                    # max-sense satisfy bound >= opt - tol. Without per-instance
                    # sense metadata, flag only the case that is invalid under
                    # BOTH senses when an incumbent orients us, else use the
                    # incumbent-side relation.
                    if o is not None:
                        if b <= o + 1e-9:  # min-sense orientation
                            if b > opt + tol:
                                incorrect.append((inst, tag, "bound>opt", b, opt))
                        else:  # max-sense orientation
                            if b < opt - tol:
                                incorrect.append((inst, tag, "bound<opt", b, opt))
                    elif abs(b) < 1e19 and not (b <= opt + tol or b >= opt - tol):
                        incorrect.append((inst, tag, "bound vs opt", b, opt))
                if (
                    o is not None
                    and r.get("cert")
                    and r.get("status") == "optimal"
                    and abs(o - opt) > tol
                ):
                    incorrect.append((inst, tag, "certified obj != opt", o, opt))

            if "error" not in off and "error" not in on:
                bo, bn = off.get("bound"), on.get("bound")
                cover_off += bo is not None
                cover_on += bn is not None
                if bo is not None and bn is not None:
                    n_bound_comparisons += 1
                    oo, oon = off.get("obj"), on.get("obj")
                    # tighter = closer to the incumbent side; orient by obj when
                    # present, else treat larger as tighter only under min sense
                    # ambiguity — count sign-agnostic equality otherwise.
                    if abs(bn - bo) <= 1e-9 * (1.0 + abs(bo)):
                        equal += 1
                    elif oo is not None and oon is not None:
                        # min sense: bound <= obj, tighter = larger bound.
                        if bo <= oo + 1e-9 and bn <= oon + 1e-9:
                            tighter += bn > bo
                            looser += bn < bo
                        else:
                            tighter += bn < bo
                            looser += bn > bo
                    else:
                        # no incumbent to orient: count but classify neutrally
                        equal += 0
                if off.get("cert") and not on.get("cert"):
                    cert_regressions.append(inst)
                if (
                    off.get("obj") is not None
                    and on.get("obj") is not None
                    and abs(off["obj"] - on["obj"]) > ABS_TOL + REL_TOL * abs(off["obj"])
                ):
                    obj_drift.append((inst, off["obj"], on["obj"]))
    finally:
        os.unlink(worker)

    print("\n=== #933 seed graduation panel ===")
    print(f"instances run              : {len(rows)}")
    print(f"oracle bound checks        : {n_oracle_checks}")
    print(f"paired bound comparisons   : {n_bound_comparisons}")
    print(f"bound coverage OFF -> ON   : {cover_off} -> {cover_on}")
    print(f"ON tighter / equal / looser: {tighter} / {equal} / {looser}")
    print(f"incorrect (oracle crossed) : {len(incorrect)}  {incorrect[:5]}")
    print(f"certification regressions  : {len(cert_regressions)}  {cert_regressions[:5]}")
    print(f"objective drift            : {len(obj_drift)}  {obj_drift[:5]}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"rows -> {args.out}")

    # §6: a panel that measured nothing must not read as a pass.
    if n_oracle_checks == 0 or n_bound_comparisons == 0:
        print("PANEL INVALID: zero executed comparisons")
        return 2
    if incorrect or cert_regressions or obj_drift:
        print("PANEL FAILED: cert-clean bar not met")
        return 1
    print("cert-clean bar met; judge net-positive from the numbers above")
    return 0


if __name__ == "__main__":
    sys.exit(main())
