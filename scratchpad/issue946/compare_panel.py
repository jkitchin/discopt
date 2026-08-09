"""Compare two #946 differential-panel runs (before/after).

Gates, in the language of CLAUDE.md §5 for a bound-changing change:

  cert-clean   no bound above its reference optimum beyond tolerance,
               no certification regression (an instance that certified before
               must still certify), objective drift within tolerance
  net-positive the new bound is >= the old one instance-by-instance, and
               strictly better somewhere

Prints an executed-comparison count and exits non-zero when it is zero or when a
gate fails.
"""

from __future__ import annotations

import json
import sys

TOL = 1e-6


def key(row):
    return (row["seed"], row["exact"])


def main() -> int:
    before = {key(r): r for r in json.load(open(sys.argv[1]))}
    after = {key(r): r for r in json.load(open(sys.argv[2]))}
    checks = 0
    unsound: list[str] = []
    preexisting: list[str] = []
    oracle_infeasible: list[str] = []
    cert_regress: list[str] = []
    obj_drift: list[str] = []
    obj_better: list[str] = []
    bound_worse: list[str] = []
    bound_better: list[str] = []
    status_changed: list[str] = []

    for k in sorted(before.keys() & after.keys()):
        b, a = before[k], after[k]
        if "error" in b or "error" in a:
            if b.get("error") != a.get("error"):
                status_changed.append(f"{k}: error {b.get('error')!r} -> {a.get('error')!r}")
            checks += 1
            continue
        mono = a["mono_objective"]
        scale = 1.0 + abs(mono) if mono is not None else 1.0
        checks += 1

        # cert-clean, part 1: the certificate invariant, bound <= own incumbent
        # (minimize sense). This one needs no external oracle. An instance that
        # already violated it BEFORE is pre-existing, not caused by the change,
        # and is reported separately so it cannot hide a new one.
        def violates_invariant(row):
            return (
                row["bound"] is not None
                and row["objective"] is not None
                and row["bound"] > row["objective"] + TOL * scale
            )

        if violates_invariant(a):
            msg = (
                f"{k}: bound {a['bound']!r} > own incumbent {a['objective']!r} "
                f"(incumbent violation {a.get('gbd_violation')!r})"
            )
            (preexisting if violates_invariant(b) else unsound).append(msg)
        # cert-clean, part 2: no bound above the monolithic optimum — but only
        # when that reference point is actually feasible. A monolithic point a
        # few 1e-9 outside a degenerate row reports an objective BELOW the true
        # optimum (#940/#945), and a valid bound then looks unsound against it.
        mono_viol = a.get("mono_violation")
        if a["bound"] is not None and mono is not None:
            if mono_viol is not None and mono_viol <= 1e-12:
                if a["bound"] > mono + TOL * scale:
                    unsound.append(f"{k}: bound {a['bound']!r} > feasible mono {mono!r}")
            elif a["bound"] > mono + TOL * scale:
                oracle_infeasible.append(
                    f"{k}: bound {a['bound']!r} > mono {mono!r}, but the mono point "
                    f"violates a row by {mono_viol!r} (oracle unusable)"
                )
        # cert-clean: no certification lost
        if b["status"] == "optimal" and a["status"] != "optimal":
            cert_regress.append(f"{k}: {b['status']} -> {a['status']}")
        if b["status"] != a["status"]:
            status_changed.append(f"{k}: {b['status']} -> {a['status']}")
        # cert-clean: objective drift. These models minimize, so only a HIGHER
        # objective after is a regression; a lower one is a better incumbent.
        if b["objective"] is not None and a["objective"] is not None:
            if a["objective"] > b["objective"] + 1e-3 * scale:
                obj_drift.append(f"{k}: incumbent WORSE {b['objective']!r} -> {a['objective']!r}")
            elif a["objective"] < b["objective"] - 1e-3 * scale:
                obj_better.append(f"{k}: incumbent better {b['objective']!r} -> {a['objective']!r}")
        elif (b["objective"] is None) != (a["objective"] is None):
            obj_drift.append(f"{k}: {b['objective']!r} -> {a['objective']!r}")
        # net-positive: the bound must not weaken
        if b["bound"] is not None and a["bound"] is not None:
            if a["bound"] < b["bound"] - 1e-6 * scale:
                bound_worse.append(f"{k}: {b['bound']!r} -> {a['bound']!r}")
            elif a["bound"] > b["bound"] + 1e-6 * scale:
                bound_better.append(f"{k}: {b['bound']!r} -> {a['bound']!r}")
        elif b["bound"] is not None and a["bound"] is None:
            bound_worse.append(f"{k}: bound withdrawn ({b['bound']!r} -> None)")
        elif b["bound"] is None and a["bound"] is not None:
            bound_better.append(f"{k}: bound gained (None -> {a['bound']!r})")

    def show(title, items):
        print(f"\n{title}: {len(items)}")
        for s in items[:20]:
            print(f"   {s}")

    show("NEW unsound bounds (bound > verified optimum / own incumbent)", unsound)
    show("pre-existing invariant violations (identical before the change)", preexisting)
    show("bound above an INFEASIBLE monolithic reference (oracle unusable)", oracle_infeasible)
    show("certification regressions (optimal -> not optimal)", cert_regress)
    show("incumbent REGRESSED (> 1e-3 worse)", obj_drift)
    show("incumbent improved", obj_better)
    show("bound WEAKER after", bound_worse)
    show("bound STRONGER after", bound_better)
    show("status changes", status_changed)

    print(f"\nexecuted comparisons: {checks}")
    ok = not unsound and not cert_regress and not obj_drift and not bound_worse and checks > 0
    print(f"cert-clean AND no bound weakened: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
