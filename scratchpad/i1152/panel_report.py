"""#1152 §5 panel report — soundness, certification, bound quality, punctuality.

Reads the JSONL from ``panel.py`` and answers the two §5 bars:

  * **cert-clean** — no dual bound above the instance's reference optimum in either
    arm, no ``bound > objective`` inside a run, no certified instance losing its
    certificate or changing its certified objective;
  * **net-positive** — the bound/punctuality differential, ON vs OFF.

The reference optimum is the tightest UPPER bound on the true optimum available:
``known_optima.toml`` where the instance is listed, otherwise the best feasible
objective any run of any arm returned (a feasible incumbent bounds the optimum,
whatever the run's status). Prints an executed-comparison count and exits non-zero
when it is zero (§6) or when a soundness violation is found.
"""

from __future__ import annotations

import json
import math
import sys
import tomllib
from collections import defaultdict

TOL_ABS = 1e-6
TOL_REL = 1e-4


def _tol(v: float) -> float:
    return TOL_ABS + TOL_REL * max(1.0, abs(v))


def main(path: str) -> int:
    recs = [json.loads(line) for line in open(path) if line.strip() and not line.startswith("#")]
    if not recs:
        print("no records")
        return 1

    with open("python/tests/data/known_optima.toml", "rb") as fh:
        known = tomllib.load(fh)

    # Reference optimum per instance: the tightest upper bound we can justify.
    ref: dict[str, float] = {}
    sense: dict[str, str] = {}
    for r in recs:
        sense[r["instance"]] = r["sense"]
        if r["objective"] is not None and math.isfinite(r["objective"]):
            cur = ref.get(r["instance"])
            better = (
                r["objective"]
                if cur is None
                else (min(cur, r["objective"]) if r["sense"] == "min" else max(cur, r["objective"]))
            )
            ref[r["instance"]] = better
    for name, entry in known.items():
        if isinstance(entry, dict) and "optimum" in entry:
            o = float(entry["optimum"])
            if name in sense:
                cur = ref.get(name)
                if cur is None:
                    ref[name] = o
                else:
                    ref[name] = min(cur, o) if sense[name] == "min" else max(cur, o)

    n_cmp = 0
    violations: list[str] = []
    for r in recs:
        b = r["bound"]
        if b is None or not math.isfinite(b):
            continue
        o = r["objective"]
        if o is not None and math.isfinite(o):
            n_cmp += 1
            bad = (b > o + _tol(o)) if r["sense"] == "min" else (b < o - _tol(o))
            if bad:
                violations.append(
                    f"BOUND-CROSSES-INCUMBENT {r['instance']} T={r['time_limit']} "
                    f"{r['arm']} bound={b} objective={o}"
                )
        ro = ref.get(r["instance"])
        if ro is not None and math.isfinite(ro):
            n_cmp += 1
            bad = (b > ro + _tol(ro)) if r["sense"] == "min" else (b < ro - _tol(ro))
            if bad:
                violations.append(
                    f"BOUND-CROSSES-REFERENCE {r['instance']} T={r['time_limit']} "
                    f"{r['arm']} bound={b} ref_opt={ro}"
                )

    # Pair the arms per (instance, T, rep).
    pairs: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        pairs[(r["instance"], r["time_limit"], r["rep"])][r["arm"]] = r

    bound_gained, bound_lost, tighter, looser, cert_lost, cert_obj_changed = [], [], [], [], [], []
    ratio_off, ratio_on = [], []
    for key, arms in sorted(pairs.items()):
        off, on = arms.get("off"), arms.get("on")
        if off is None or on is None:
            continue
        n_cmp += 1
        ratio_off.append(off["ratio"])
        ratio_on.append(on["ratio"])
        bo, bn = off["bound"], on["bound"]
        tag = f"{key[0]} T={key[1]} rep={key[2]}"
        if bo is None and bn is not None:
            bound_gained.append(f"{tag}: None -> {bn}")
        elif bo is not None and bn is None:
            bound_lost.append(f"{tag}: {bo} -> None")
        elif bo is not None and bn is not None:
            d = bn - bo if off["sense"] == "min" else bo - bn
            if d > _tol(bo):
                tighter.append(f"{tag}: {bo} -> {bn}")
            elif d < -_tol(bo):
                looser.append(f"{tag}: {bo} -> {bn}")
        if off["gap_certified"] and not on["gap_certified"]:
            cert_lost.append(f"{tag}: certified OFF, not ON")
        if (
            off["status"] == "optimal"
            and on["status"] == "optimal"
            and off["objective"] is not None
            and on["objective"] is not None
            and abs(off["objective"] - on["objective"]) > _tol(off["objective"])
        ):
            cert_obj_changed.append(f"{tag}: {off['objective']} -> {on['objective']}")

    def _p(title, items):
        print(f"\n{title}: {len(items)}")
        for it in items:
            print(f"   {it}")

    print(f"records={len(recs)} pairs={len(ratio_off)} comparisons={n_cmp}")
    _p("SOUNDNESS VIOLATIONS", violations)
    _p("certified-status lost (ON)", cert_lost)
    _p("certified objective changed", cert_obj_changed)
    _p("bound RECOVERED (None -> finite)", bound_gained)
    _p("bound LOST (finite -> None)", bound_lost)
    _p("bound TIGHTER", tighter)
    _p("bound LOOSER", looser)
    if ratio_off:
        print(
            f"\nwall/time_limit  OFF: mean={sum(ratio_off) / len(ratio_off):.3f} "
            f"max={max(ratio_off):.3f} over_1.25x={sum(1 for x in ratio_off if x > 1.25)}"
        )
        print(
            f"wall/time_limit   ON: mean={sum(ratio_on) / len(ratio_on):.3f} "
            f"max={max(ratio_on):.3f} over_1.25x={sum(1 for x in ratio_on if x > 1.25)}"
        )
    print(f"\n# comparisons={n_cmp}")
    if n_cmp == 0:
        return 1
    return 2 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
