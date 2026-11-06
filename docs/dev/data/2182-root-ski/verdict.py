#!/usr/bin/env python3
"""Verdict for the pre-registered §5 panel of ``DISCOPT_GDP_CONFIG_ROOT_SKI`` (#2182).

Applies exactly the bars in ``prereg.md`` (same directory), committed before any
panel row existed:

    python docs/dev/data/2182-root-ski/verdict.py docs/dev/data/2182-root-ski

Reads ``gdp_parity_120s.jsonl`` (``issue1917_gdp_flag_panel.py`` rows) and
``corpus_60s_2rep.jsonl`` (``flag_onoff_corpus_panel.py`` rows). Exits 0 on GRADUATE,
1 on RETIRE or a C violation, 2 on a void run (a V failure).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "discopt_benchmarks"))
sys.path.insert(0, str(REPO / "discopt_benchmarks" / "scripts"))

REL = 1e-4
REPS = (0, 1)


def sgm(walls, shift=1.0):
    return math.exp(statistics.mean(math.log(w + shift) for w in walls)) - shift


def gmean(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


def better(a, b, minimize):
    """``a`` strictly better than ``b`` beyond the relative tolerance."""
    tol = REL * max(1.0, abs(b))
    return a < b - tol if minimize else a > b + tol


def main(d: Path) -> int:
    import issue1917_gdp_flag_panel as P

    void, cviol, nfail, notes = [], [], [], []
    n_assert = 0

    # ---------------- panel A ----------------
    rows = [json.loads(ln) for ln in (d / "gdp_parity_120s.jsonl").open() if ln.strip()]
    rows = [r for r in rows if r.get("phase") == "solve"]
    A: dict = defaultdict(dict)  # (instance, method) -> {(arm, rep): row}
    for r in rows:
        A[(r["instance"], r["method"])][(r["arm"], r["rep"])] = r
    if len(rows) != 192:
        void.append(f"V2: panel A has {len(rows)} solve rows, expected 192")
    wt = str(REPO / "python")
    bad_path = [r for r in rows if not str(r.get("discopt", "")).startswith(wt)]
    if bad_path:
        void.append(f"V1: {len(bad_path)} panel A rows loaded discopt outside {wt}")
    if not any(r["arm"] == "on" and r.get("in_pyomo") for r in rows):
        void.append("V3: no ON incumbent re-checked in Pyomo in panel A")

    oracle = P.gb_oracle()
    for r in rows:
        n_assert += 1
        if r.get("verdict") == "incorrect":
            cviol.append(f"C1 A {r['instance']} {r['method']} {r['arm']} rep{r['rep']}")
        if P._bound_crosses(r, oracle.get(r["instance"])):
            cviol.append(f"C2 A {r['instance']} {r['method']} {r['arm']} rep{r['rep']}")

    def solved(r):
        return r is not None and r.get("verdict") == "solved"

    def has_inc(r):
        return r is not None and r.get("objective") is not None

    def is_err(r):
        return r is None or r.get("verdict") == "error"

    cert_both, primal_on_better, primal_on_worse = [], 0, 0
    for cell, runs in sorted(A.items()):
        g = {(a, k): runs.get((a, k)) for a in ("off", "on") for k in REPS}
        for k in REPS:
            if solved(g[("off", k)]) and not solved(g[("on", k)]):
                off_stable = all(solved(g[("off", j)]) for j in REPS)
                (cviol if off_stable else notes).append(
                    f"C3 A {cell} rep{k}: solved OFF, not ON"
                    + ("" if off_stable else " (OFF itself flips across reps: noise)")
                )
        if all(is_err(g[("on", k)]) for k in REPS) and not any(is_err(g[("off", k)]) for k in REPS):
            cviol.append(f"C4 A {cell}: ON errors in both reps")
        if all(has_inc(g[("off", k)]) for k in REPS) and not any(
            has_inc(g[("on", k)]) for k in REPS
        ):
            nfail.append(f"N2 A {cell}: incumbent lost")
        if all(solved(g[(a, k)]) for a in ("off", "on") for k in REPS):
            cert_both.append(cell)
        elif all(has_inc(g[(a, k)]) and not solved(g[(a, k)]) for a in ("off", "on") for k in REPS):
            mn = g[("off", 0)].get("minimize", True)
            ob = [better(g[("on", k)]["objective"], g[("off", k)]["objective"], mn) for k in REPS]
            ow = [better(g[("off", k)]["objective"], g[("on", k)]["objective"], mn) for k in REPS]
            primal_on_better += all(ob)
            primal_on_worse += all(ow)
    if primal_on_better < primal_on_worse:
        nfail.append(f"N3 A: ON better in {primal_on_better} cells, worse in {primal_on_worse}")

    print(
        f"panel A: {len(rows)} solve rows, {len(A)} cells, "
        f"{len(cert_both)} certified by both arms in both reps"
    )
    for k in REPS:
        if not cert_both:
            nfail.append("N1 A: no cell certified by both arms in both reps")
            break
        w_off = [A[c][("off", k)]["wall"] for c in cert_both]
        w_on = [A[c][("on", k)]["wall"] for c in cert_both]
        ratio = sgm(w_on) / sgm(w_off)
        print(
            f"  N1 rep{k}: SGM wall (shift 1 s) OFF {sgm(w_off):.3f} s "
            f"ON {sgm(w_on):.3f} s ratio {ratio:.3f}"
        )
        if not ratio <= 0.95:
            nfail.append(f"N1 A rep{k}: SGM wall ratio {ratio:.3f} > 0.95")
    if cert_both:

        def med_nodes(c, a):
            return statistics.median((A[c][(a, k)].get("node_count") or 0) for k in REPS)

        gm = gmean([(med_nodes(c, "on") + 1) / (med_nodes(c, "off") + 1) for c in cert_both])
        print(f"  N4 A: node ratio geomean {gm:.3f}")
        if gm > 1.05:
            nfail.append(f"N4 A: node ratio geomean {gm:.3f} > 1.05")
    print(f"  N3 A: ON better in {primal_on_better} cells, worse in {primal_on_worse}")

    # ---------------- panel B ----------------
    brows = [json.loads(ln) for ln in (d / "corpus_60s_2rep.jsonl").open() if ln.strip()]
    if len(brows) != 148:  # one row per (instance, rep), each holding both arms
        void.append(
            f"V2: panel B has {len(brows)} rows ({2 * len(brows)} solves), expected 148 (296)"
        )
    bad_b = [
        r
        for r in brows
        for a in ("off", "on")
        if not str(r.get(a, {}).get("discopt_file", "")).startswith(wt)
    ]
    if bad_b:
        void.append(f"V1: {len(bad_b)} panel B arms loaded discopt outside {wt}")
    if not any(r.get("on", {}).get("feasible") is True for r in brows):
        void.append("V3: no ON incumbent feasibility-checked in panel B")
    B: dict = defaultdict(dict)
    for r in brows:
        n_assert += 1
        B[r["name"]][r["rep"]] = r
        if r.get("problems"):
            cviol.append(f"C1 B {r['name']} rep{r['rep']}: {r['problems']}")
    certb = []
    for name, reps in sorted(B.items()):
        for k in REPS:
            r = reps.get(k)
            if r and r["off"].get("cert") and not r["on"].get("cert"):
                off_stable = all(reps.get(j) and reps[j]["off"].get("cert") for j in REPS)
                (cviol if off_stable else notes).append(
                    f"C3 B {name} rep{k}: cert OFF, not ON"
                    + ("" if off_stable else " (OFF itself flips across reps: noise)")
                )
        if all(reps.get(k) for k in REPS):
            inc = {a: [reps[k][a].get("obj") is not None for k in REPS] for a in ("off", "on")}
            if all(inc["off"]) and not any(inc["on"]):
                nfail.append(f"N2 B {name}: incumbent lost")
            if all(reps[k][a].get("cert") for a in ("off", "on") for k in REPS):
                certb.append(name)
    for k in REPS:
        w_off = [B[n][k]["off"]["wall"] for n in B if k in B[n]]
        w_on = [B[n][k]["on"]["wall"] for n in B if k in B[n]]
        ratio = sgm(w_on) / sgm(w_off)
        print(
            f"panel B rep{k}: {len(w_off)} instances, SGM wall "
            f"OFF {sgm(w_off):.3f} ON {sgm(w_on):.3f} ratio {ratio:.3f}"
        )
        if not ratio <= 1.05:
            nfail.append(f"N5 B rep{k}: SGM wall ratio {ratio:.3f} > 1.05")
    if certb:
        gm = gmean(
            [
                (statistics.median(B[n][k]["on"].get("nodes") or 0 for k in REPS) + 1)
                / (statistics.median(B[n][k]["off"].get("nodes") or 0 for k in REPS) + 1)
                for n in certb
            ]
        )
        print(f"  N4 B: node ratio geomean {gm:.3f} over {len(certb)} certified-by-both")
        if gm > 1.05:
            nfail.append(f"N4 B: node ratio geomean {gm:.3f} > 1.05")

    for label, items in (
        ("VOID", void),
        ("C violations", cviol),
        ("N failures", nfail),
        ("notes", notes),
    ):
        print(f"{label}: {len(items)}")
        for x in items:
            print("   ", x)
    print(f"executed-assertions: {n_assert}")
    if n_assert == 0:
        return 2
    if void:
        print("VERDICT: VOID (K2)")
        return 2
    if cviol:
        print("VERDICT: K1 (C violation) -- flag stays OFF")
        return 1
    if nfail:
        print("VERDICT: RETIRE (C holds, N fails)")
        return 1
    print("VERDICT: GRADUATE (V, C and N hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent))
