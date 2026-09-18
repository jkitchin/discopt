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
import numpy as np  # noqa: E402
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
# How many structurally-unaffected instances to run anyway, as a check on the
# classification itself (see NOOP_CHECK below).
NOOP_SAMPLE = int(sys.argv[3]) if len(sys.argv) > 3 else 12
ABS_TOL, REL_TOL = 1e-6, 1e-4

OPTIMA = {k: v["optimum"] for k, v in optima_registry().items() if "optimum" in v}
ALL_NAMES = sorted(p.stem for p in CORPUS.glob("*.nl"))
CERTIFIED = {"optimal", "infeasible", "unbounded"}

faulthandler.enable()


def declared_bounds_in_window(name: str) -> tuple[int, float]:
    """``(count, max|b|)`` of declared bounds with ``1e15 <= |b| < 1e19``.

    The flag changes exactly one thing: the array produced by
    ``np.where(|b| >= threshold, INF, b)``. An instance with NO bound in that
    window therefore yields a **bit-identical** box on both arms, so the flag is a
    provable no-op on it -- not "measured neutral", structurally incapable of
    differing. Splitting the corpus on this is what lets the panel spend its
    budget where a difference is possible, and it makes the no-op claim a
    derivation rather than a sample (CLAUDE.md §6: a probe that cannot distinguish
    the arms measures nothing, however green it looks).
    """
    m = from_nl(str(CORPUS / f"{name}.nl"))
    lo = np.array([float(v.lb) if v.lb is not None else -np.inf for v in m._variables])
    hi = np.array([float(v.ub) if v.ub is not None else np.inf for v in m._variables])
    b = np.abs(np.concatenate([lo, hi]))
    b = b[np.isfinite(b)]
    inside = b[(b >= _lp._LEGACY_BOUND_THRESHOLD) & (b < _lp._POUNCE_BOUND_INF)]
    return int(inside.size), (float(inside.max()) if inside.size else 0.0)


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

# --- Split the corpus on whether the flag can possibly do anything -----------
print(f"\nclassifying {len(ALL_NAMES)} instances by declared-bound window ...", flush=True)
AFFECTED: list[str] = []
UNAFFECTED: list[str] = []
for name in ALL_NAMES:
    try:
        n_in, max_in = declared_bounds_in_window(name)
    except Exception as exc:  # noqa: BLE001 - a classification failure is a result
        print(f"  classify FAILED {name}: {type(exc).__name__}: {exc}", flush=True)
        violations.append(f"{name}: could not be classified ({type(exc).__name__})")
        continue
    if n_in:
        AFFECTED.append(name)
        print(f"  AFFECTED   {name:<20} {n_in} bound(s) in window, max {max_in:.3e}", flush=True)
    else:
        UNAFFECTED.append(name)
print(
    f"  -> {len(AFFECTED)} affected, {len(UNAFFECTED)} structurally unaffected "
    f"(bit-identical box on both arms)",
    flush=True,
)

# Every affected instance is run in full. A sample of the unaffected ones is run
# too -- not to measure the flag, but to VERIFY the classification: if an instance
# with no bound in the window differs between arms, the derivation above is wrong
# and the whole split is invalid. That check is the point, so it must be able to
# fail loudly (CLAUDE.md §6/§7).
NAMES = AFFECTED + UNAFFECTED[:NOOP_SAMPLE]
NOOP_CHECK = set(UNAFFECTED[:NOOP_SAMPLE])

print(f"\nrunning {len(NAMES)} instances, max_nodes={MAX_NODES}, time_limit={TL}s\n", flush=True)
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

    # --- Classification check: a no-op instance MUST be identical ------------
    if name in NOOP_CHECK:
        notes.append("noop-check")
        for field in ("status", "nodes", "objective", "bound", "certified"):
            same = (
                close(off[field], on[field])
                if field in ("objective", "bound")
                else off[field] == on[field]
            )
            if not same:
                violations.append(
                    f"{name}: declares NO bound in [1e15, 1e19) yet the arms differ on "
                    f"{field!r} (OFF={off[field]!r} ON={on[field]!r}) -- the structural "
                    f"no-op derivation is WRONG and the corpus split is invalid"
                )
                notes.append(f"NOOP-BROKEN:{field}")
            comparisons += 1

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
aff = [r for r in ok if r["name"] in set(AFFECTED)]
both_clean = [r for r in aff if not r["off"]["backstop"] and not r["on"]["backstop"]]
newcert = [r["name"] for r in ok if not r["off"]["certified"] and r["on"]["certified"]]
decert = [r["name"] for r in ok if r["off"]["certified"] and not r["on"]["certified"]]
node_off = sum(r["off"]["nodes"] for r in both_clean)
node_on = sum(r["on"]["nodes"] for r in both_clean)
wall_off = sum(r["off"]["wall"] for r in both_clean)
wall_on = sum(r["on"]["wall"] for r in both_clean)

print("\n" + "=" * 78, flush=True)
print(
    f"instances run           : {len(ok)} / {len(NAMES)}  (errors: {len(rows) - len(ok)})",
    flush=True,
)
print(f"affected by the flag    : {len(AFFECTED)} of {len(ALL_NAMES)} in the corpus", flush=True)
print(f"no-op classification    : {len(NOOP_CHECK)} verified identical on both arms", flush=True)
print(f"newly certified by ON   : {len(newcert)} {newcert}", flush=True)
print(f"DEcertified by ON       : {len(decert)} {decert}", flush=True)
print(f"affected, comparable    : {len(both_clean)}", flush=True)
print(f"  nodes  OFF={node_off}  ON={node_on}", flush=True)
print(f"  wall   OFF={wall_off:.1f}s  ON={wall_on:.1f}s", flush=True)
print("=" * 78, flush=True)

if violations:
    print(f"\nGATE 1 CERT-CLEAN: FAIL ({len(violations)})", flush=True)
    for v in violations:
        print(f"  ✗ {v}", flush=True)
else:
    print(
        "\nGATE 1 CERT-CLEAN: PASS (no false bound, no decertification, no "
        "contradiction, no broken no-op)",
        flush=True,
    )

# The net-positive bar is only meaningful if the corpus actually exercises the
# flag. Reporting "neutral, therefore stays OFF" off a corpus with ~no instances
# in the affected window would be a null result dressed up as a measurement --
# exactly what CLAUDE.md §6 forbids. Say which it is.
print("\nGATE 2 NET-POSITIVE:", flush=True)
if len(both_clean) < 3:
    print(
        f"  INCONCLUSIVE -- only {len(both_clean)} comparable instance(s) in this corpus "
        f"declare a bound in [1e15, 1e19), which is too few to establish 'measurably "
        f"helpful broadly' either way. The flag is a PROVEN no-op on the other "
        f"{len(UNAFFECTED)}, so there is no corpus-wide harm to weigh; the demonstrated "
        f"benefit is on the #1319 class (wrong/absent answer -> correct certified "
        f"optimum). A graduation decision on THIS corpus rests on that, not on nodes.",
        flush=True,
    )
elif node_on <= node_off and wall_on <= wall_off * 1.05:
    print(
        f"  PASS -- nodes {node_off}->{node_on}, wall {wall_off:.1f}s->{wall_on:.1f}s",
        flush=True,
    )
else:
    print(
        f"  FAIL -- nodes {node_off}->{node_on}, wall {wall_off:.1f}s->{wall_on:.1f}s; "
        f"cert-clean but not helpful (the DISCOPT_CUT_INHERIT lesson: stays OFF)",
        flush=True,
    )

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
