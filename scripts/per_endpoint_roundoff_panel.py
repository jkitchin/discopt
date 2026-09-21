"""CLAUDE.md §5 differential panel for the #1415 per-endpoint round-off bound.

The change is **bound-changing**: it narrows an over-estimated round-off widening
in two square rules, so every node box those rules touch can come out tighter.
Direction is the sound one (the new bound is the old bound restricted to the terms
each endpoint was actually differenced from, so it is never wider and never
under-states that endpoint's own error), but "sound in direction" is not the bar --
the panel is.

Arms, interleaved WITHIN each instance and alternating in order so a machine that
gets busier mid-run cannot systematically favour one (CLAUDE.md §9):

    base   ``nonlinear_bound_tightening.py`` at HEAD (one lumped slack per interval)
    fix    this branch (one slack per endpoint)

Each arm runs in its own subprocess, which asserts the #1415 marker present/absent
in the module it actually loaded before solving anything (§8).

Gate 1 -- CERT-CLEAN (soundness; any failure kills the change):
  * no ``fix``-arm dual bound above the instance's reference optimum,
  * no certification regression (certified in ``base`` must stay certified),
  * no status contradiction between arms,
  * objective agreement within tolerance where both arms report one.

Gate 2 -- NET-POSITIVE: nodes / wall / bound, over the corpus.

Usage::

    git show <base-ref>:python/discopt/_relax/nonlinear_bound_tightening.py > /tmp/base.py
    cp python/discopt/_relax/nonlinear_bound_tightening.py /tmp/fix.py
    python -u scripts/per_endpoint_roundoff_panel.py . python/tests/data/minlplib_nl \
        /tmp/base.py /tmp/fix.py [max_nodes] [time_limit]

The panel copies each arm's file over the working tree's module between solves and
restores the ``fix`` arm in a ``finally``, so do NOT commit while it is running --
the file on disk is whichever arm is mid-solve.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(sys.argv[1])
CORPUS = Path(sys.argv[2])
ARMS = {"base": Path(sys.argv[3]), "fix": Path(sys.argv[4])}
MAX_NODES = int(sys.argv[5]) if len(sys.argv) > 5 else 300
TL = float(sys.argv[6]) if len(sys.argv) > 6 else 30.0

TARGET = REPO / "python" / "discopt" / "_relax" / "nonlinear_bound_tightening.py"
RUNNER = Path(__file__).with_name("_per_endpoint_roundoff_solve_one.py")
ABS_TOL, REL_TOL = 1e-6, 1e-4
CERTIFIED = {"optimal", "infeasible", "unbounded"}

sys.path.insert(0, str(REPO / "python" / "tests"))
from _optima import optima_registry  # noqa: E402

OPTIMA = {k: v["optimum"] for k, v in optima_registry().items() if "optimum" in v}
NAMES = sorted(p.stem for p in CORPUS.glob("*.nl"))

# §8: the two arm files must actually differ, or the panel compares a change
# against itself and every green row means nothing.
base_src, fix_src = ARMS["base"].read_text(), ARMS["fix"].read_text()
assert base_src != fix_src, "the two arm files are identical -- nothing to compare"
assert "#1415" in fix_src and "#1415" not in base_src, "arm files are not marked as expected"
print(f"arms differ: base {len(base_src)}B, fix {len(fix_src)}B, marker split OK", flush=True)


def run(name: str, arm: str) -> dict:
    shutil.copyfile(ARMS[arm], TARGET)
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(RUNNER),
            str(CORPUS / f"{name}.nl"),
            arm,
            str(MAX_NODES),
            str(TL),
        ],
        capture_output=True,
        text=True,
        timeout=TL + 300.0,
    )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            out = json.loads(line)
            if "error" in out:
                return {"crash": out["error"]}
            return out
    return {"crash": f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"}


def close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= ABS_TOL + REL_TOL * max(1.0, abs(a), abs(b))


rows: list[dict] = []
violations: list[str] = []
comparisons = 0  # CLAUDE.md §6: executed-assertion count.

hdr = (
    f"{'instance':<20} {'base status':<13} {'fix status':<13} "
    f"{'baseNd':>7} {'fixNd':>7} {'baseS':>7} {'fixS':>7} note"
)
print(f"\n{len(NAMES)} instances, max_nodes={MAX_NODES}, time_limit={TL}s\n", flush=True)
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

t_start = time.perf_counter()
try:
    for i, name in enumerate(NAMES, 1):
        if i % 2:
            base, fix = run(name, "base"), run(name, "fix")
        else:
            fix, base = run(name, "fix"), run(name, "base")

        if "crash" in base or "crash" in fix:
            comparisons += 1
            note = f"CRASH base={base.get('crash', '-')[:70]} fix={fix.get('crash', '-')[:70]}"
            # A crash on ONE arm only is a regression signal; on both it is the
            # instance, not the change.
            if ("crash" in fix) and ("crash" not in base):
                violations.append(f"{name}: fix arm crashed, base did not -- {fix['crash'][:160]}")
            print(f"{name:<20} {note}", flush=True)
            rows.append({"name": name, "base": base, "fix": fix})
            continue

        notes = []
        opt = OPTIMA.get(name)
        sense_min = True  # .nl models here are normalized to min by the reader

        # --- Gate 1: soundness ------------------------------------------------
        if opt is not None and fix.get("bound") is not None and not fix["backstop"]:
            comparisons += 1
            slack = ABS_TOL + REL_TOL * max(1.0, abs(opt))
            if sense_min and fix["bound"] > opt + slack:
                violations.append(
                    f"{name}: FIX dual bound {fix['bound']!r} above reference optimum {opt!r}"
                )
                notes.append("BOUND>OPT")

        comparisons += 1
        if base["status"] in CERTIFIED and fix["status"] in CERTIFIED:
            contradiction = {base["status"], fix["status"]} in (
                {"optimal", "infeasible"},
                {"optimal", "unbounded"},
                {"infeasible", "unbounded"},
            )
            if contradiction:
                violations.append(
                    f"{name}: status contradiction base={base['status']} fix={fix['status']}"
                )
                notes.append("STATUS")

        comparisons += 1
        if base["certified"] and not fix["certified"] and not fix["backstop"]:
            violations.append(f"{name}: certification regression (base certified, fix not)")
            notes.append("CERT-REGRESS")

        comparisons += 1
        if base["objective"] is not None and fix["objective"] is not None:
            if not close(base["objective"], fix["objective"]):
                violations.append(
                    f"{name}: objective drift base={base['objective']!r} fix={fix['objective']!r}"
                )
                notes.append("OBJ-DRIFT")

        if not close(base.get("bound"), fix.get("bound")):
            notes.append("bound-moved")
        if base["nodes"] != fix["nodes"]:
            notes.append("nodes-moved")

        rows.append({"name": name, "base": base, "fix": fix})
        print(
            f"{name:<20} {base['status']:<13} {fix['status']:<13} "
            f"{base['nodes']:>7} {fix['nodes']:>7} "
            f"{base['wall']:>7.2f} {fix['wall']:>7.2f} {' '.join(notes)}",
            flush=True,
        )
finally:
    # Always leave the working tree on the branch under test.
    shutil.copyfile(ARMS["fix"], TARGET)

# ----------------------------------------------------------------------------- #
# Summary
# ----------------------------------------------------------------------------- #
ok = [r for r in rows if "crash" not in r["base"] and "crash" not in r["fix"]]
comparable = [r for r in ok if not r["base"]["backstop"] and not r["fix"]["backstop"]]
b_nodes = sum(r["base"]["nodes"] for r in comparable)
f_nodes = sum(r["fix"]["nodes"] for r in comparable)
b_wall = sum(r["base"]["wall"] for r in comparable)
f_wall = sum(r["fix"]["wall"] for r in comparable)
b_cert = sum(r["base"]["certified"] for r in ok)
f_cert = sum(r["fix"]["certified"] for r in ok)
moved = [r["name"] for r in comparable if r["base"]["nodes"] != r["fix"]["nodes"]]
bound_moved = [r["name"] for r in comparable if not close(r["base"]["bound"], r["fix"]["bound"])]

print(f"\n{'=' * 78}")
print(f"instances run            : {len(rows)}  (comparable: {len(comparable)})")
print(f"executed comparisons     : {comparisons}")
print(f"certified   base -> fix  : {b_cert} -> {f_cert}")
print(f"total nodes base -> fix  : {b_nodes} -> {f_nodes}")
print(f"total wall  base -> fix  : {b_wall:.1f}s -> {f_wall:.1f}s")
print(f"instances whose node count moved : {len(moved)} {moved[:12]}")
print(f"instances whose dual bound moved : {len(bound_moved)} {bound_moved[:12]}")
print(f"elapsed: {time.perf_counter() - t_start:.0f}s")
print(f"{'=' * 78}")

json.dump(rows, open("per_endpoint_roundoff_rows.json", "w"), indent=1)

if comparisons == 0:
    print("PANEL MEASURED NOTHING")
    sys.exit(2)
if violations:
    print(f"\nGATE 1 FAILED -- {len(violations)} soundness violation(s):")
    for v in violations:
        print(f"  * {v}")
    sys.exit(1)
print("\nGATE 1 (cert-clean): PASS")
sys.exit(0)
