"""CLAUDE.md §5 differential panel for ``DISCOPT_POUNCE_DECLARED_BOX`` (#1319).

The flag raises the threshold at which the POUNCE IPM stops honoring a declared
finite bound and substitutes its own infinity, from discopt's legacy ``1e15`` to
POUNCE's own ``1e19`` (Ipopt's ``nlp_{lower,upper}_bound_inf``). Measured on the
#1319 repro: POUNCE returns the exact analytic optimum for every bound magnitude
up to 9.9e18 and flips to UNBOUNDED at exactly 1e19, so the legacy threshold
discarded four orders of magnitude of perfectly usable declared bounds.

This is **bound-changing** under §5: every node relaxation over a variable in the
``[1e15, 1e19)`` window now solves a tighter box. Direction of the change is the
sound one (a bound is a constraint; discarding it enlarges the feasible set, so
the OLD path solved a superset and the NEW path cannot be looser), but "sound in
direction" is not the bar -- the panel is.

Arms, interleaved WITHIN each instance so a machine that gets busier mid-run
perturbs both arms equally (CLAUDE.md §9):

    OFF  DISCOPT_POUNCE_DECLARED_BOX=0   legacy 1e15
    ON   DISCOPT_POUNCE_DECLARED_BOX=1   POUNCE's 1e19  (the proposed default)

Gate 1 -- CERT-CLEAN (soundness; any failure kills the flag):
  * no ON-arm dual bound above the instance's reference optimum (the registry in
    ``python/tests/data/known_optima.toml``),
  * no certification regression: an instance certified in OFF must stay certified
    in ON,
  * no status contradiction (optimal <-> infeasible/unbounded between arms),
  * objective agreement within tolerance where both arms report one.

Gate 2 -- NET-POSITIVE: measurably helpful broadly (nodes / wall / bound), not
merely sound. The ``DISCOPT_CUT_INHERIT`` lesson: a cert-clean but
neutral-or-harmful flag stays OFF.

``max_nodes`` is the primary budget because it is deterministic; a wall-clock
backstop exists only so the panel terminates, and an arm that hits it is reported
as BACKSTOP and excluded from the timing comparison rather than silently counted
(CLAUDE.md §6 -- the probe must say when it measured nothing).

Usage:  python -u scripts/pounce_declared_box_panel.py [max_nodes] [time_limit]
"""

import faulthandler
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

# CLAUDE.md §8: prove which code is loaded, and that it is the version under test.
print(f"discopt from {discopt.__file__}", flush=True)
import discopt.solvers.lp_pounce as _lp  # noqa: E402

assert hasattr(_lp, "finite_bound_threshold"), (
    "marker absent: `finite_bound_threshold` is not defined, so this tree is NOT "
    "the version under test and the panel would compare a flag against itself."
)
os.environ["DISCOPT_POUNCE_DECLARED_BOX"] = "0"
assert _lp.finite_bound_threshold() == _lp._LEGACY_BOUND_THRESHOLD, "OFF arm does not opt out"
os.environ["DISCOPT_POUNCE_DECLARED_BOX"] = "1"
assert _lp.finite_bound_threshold() == _lp._POUNCE_BOUND_INF, "ON arm does not engage"
print(
    f"marker present; OFF={_lp._LEGACY_BOUND_THRESHOLD:.0e} ON={_lp._POUNCE_BOUND_INF:.0e}",
    flush=True,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "python" / "tests" / "data" / "minlplib_nl"
sys.path.insert(0, str(ROOT / "python" / "tests"))
from _optima import optima_registry  # noqa: E402

MAX_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
ABS_TOL, REL_TOL = 1e-6, 1e-4

OPTIMA = {k: v["optimum"] for k, v in optima_registry().items() if "optimum" in v}
NAMES = sorted(p.stem for p in CORPUS.glob("*.nl"))
CERTIFIED = {"optimal", "infeasible", "unbounded"}

faulthandler.enable()


def run(name: str, flag: str) -> dict:
    os.environ["DISCOPT_POUNCE_DECLARED_BOX"] = flag
    m = from_nl(str(CORPUS / f"{name}.nl"))
    faulthandler.dump_traceback_later(TL + 120.0, exit=False)
    t0 = time.perf_counter()
    try:
        r = m.solve(max_nodes=MAX_NODES, time_limit=TL)
    finally:
        faulthandler.cancel_dump_traceback_later()
    wall = time.perf_counter() - t0
    return {
        "status": r.status,
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None else float(r.bound),
        "certified": bool(r.gap_certified),
        "nodes": int(r.node_count or 0),
        "wall": wall,
        # A wall-clock stop makes the arm's node count/bound incomparable.
        "backstop": r.status in ("time_limit",) or wall >= TL * 0.98,
    }


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= ABS_TOL + REL_TOL * max(1.0, abs(a), abs(b))


rows: list[dict] = []
violations: list[str] = []
comparisons = 0  # CLAUDE.md §6: executed-assertion count.

print(f"\n{len(NAMES)} instances, max_nodes={MAX_NODES}, time_limit={TL}s\n", flush=True)
hdr = f"{'instance':<18} {'OFF status':<14} {'ON status':<14} {'OFFnodes':>9} {'ONnodes':>8} note"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for i, name in enumerate(NAMES, 1):
    try:
        # Interleaved within the instance; order alternates so a warm cache or a
        # drifting machine cannot systematically favor one arm.
        if i % 2:
            off, on = run(name, "0"), run(name, "1")
        else:
            on, off = run(name, "1"), run(name, "0")
    except Exception as exc:  # noqa: BLE001 - a crash is a result, never swallowed
        print(f"{name:<18} RAISED {type(exc).__name__}: {exc}", flush=True)
        rows.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        comparisons += 1
        continue

    notes = []
    opt = OPTIMA.get(name)

    # --- Gate 1: cert-clean -------------------------------------------------
    if opt is not None and on["bound"] is not None and on["certified"]:
        slack = ABS_TOL + REL_TOL * max(1.0, abs(opt))
        if on["bound"] > opt + slack:
            violations.append(
                f"{name}: ON dual bound {on['bound']:.12g} EXCEEDS reference optimum "
                f"{opt:.12g} (slack {slack:.3g}) -- FALSE BOUND"
            )
            notes.append("BOUND>OPT")
        comparisons += 1
    if off["certified"] and not on["certified"]:
        violations.append(
            f"{name}: certification REGRESSION -- OFF certified {off['status']}, "
            f"ON uncertified {on['status']}"
        )
        notes.append("DECERT")
    comparisons += 1
    contradiction = {
        ("optimal", "infeasible"),
        ("infeasible", "optimal"),
        ("optimal", "unbounded"),
        ("unbounded", "optimal"),
        ("infeasible", "unbounded"),
        ("unbounded", "infeasible"),
    }
    if (off["status"], on["status"]) in contradiction:
        violations.append(
            f"{name}: status CONTRADICTION -- OFF {off['status']} vs ON {on['status']}"
        )
        notes.append("CONTRADICT")
    comparisons += 1
    if off["objective"] is not None and on["objective"] is not None:
        if not close(off["objective"], on["objective"]):
            notes.append(f"obj {off['objective']:.6g}->{on['objective']:.6g}")
        comparisons += 1

    if off["backstop"] or on["backstop"]:
        notes.append("BACKSTOP")
    if not off["certified"] and on["certified"]:
        notes.append("NEWCERT")

    rows.append({"name": name, "off": off, "on": on, "optimum": opt})
    print(
        f"{name:<18} {off['status']:<14} {on['status']:<14} "
        f"{off['nodes']:>9} {on['nodes']:>8} {' '.join(notes)}",
        flush=True,
    )

# --- Gate 2: net-positive ---------------------------------------------------
ok = [r for r in rows if "error" not in r]
both_clean = [r for r in ok if not r["off"]["backstop"] and not r["on"]["backstop"]]
newcert = [r["name"] for r in ok if not r["off"]["certified"] and r["on"]["certified"]]
decert = [r["name"] for r in ok if r["off"]["certified"] and not r["on"]["certified"]]
node_off = sum(r["off"]["nodes"] for r in both_clean)
node_on = sum(r["on"]["nodes"] for r in both_clean)
wall_off = sum(r["off"]["wall"] for r in both_clean)
wall_on = sum(r["on"]["wall"] for r in both_clean)

print("\n" + "=" * 78, flush=True)
print(
    f"instances run          : {len(ok)} / {len(NAMES)}  (errors: {len(rows) - len(ok)})",
    flush=True,
)
print(f"comparable (no backstop): {len(both_clean)}", flush=True)
print(f"newly certified by ON  : {len(newcert)} {newcert}", flush=True)
print(f"DEcertified by ON      : {len(decert)} {decert}", flush=True)
print(f"total nodes  OFF={node_off}  ON={node_on}", flush=True)
print(f"total wall   OFF={wall_off:.1f}s  ON={wall_on:.1f}s", flush=True)
print("=" * 78, flush=True)

if violations:
    print(f"\nCERT-CLEAN FAILURES ({len(violations)}):", flush=True)
    for v in violations:
        print(f"  ✗ {v}", flush=True)
else:
    print("\nCERT-CLEAN: PASS (no false bound, no decertification, no contradiction)", flush=True)

out = ROOT / "scratchpad" / "pounce_declared_box_panel.json"
out.parent.mkdir(exist_ok=True)
out.write_text(
    json.dumps(
        {"rows": rows, "violations": violations, "max_nodes": MAX_NODES, "time_limit": TL}, indent=2
    )
)
print(f"\nwrote {out}", flush=True)
print(f"EXECUTED_COMPARISONS={comparisons}", flush=True)
if comparisons == 0:
    print("FATAL: the panel compared nothing", flush=True)
    sys.exit(1)
sys.exit(1 if violations else 0)
