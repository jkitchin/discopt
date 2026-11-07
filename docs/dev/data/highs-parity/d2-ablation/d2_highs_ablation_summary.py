#!/usr/bin/env python
"""Summarise d2_highs_ablation JSONL rows: per-instance matrix, critical components, ranking.

A component is *critical* for an instance when its ablation arm (threads=1) either fails
to certify in the 20 s budget, or certifies with >= 10x the baseline node count (the
task's two bars). Ties with baseline node 0/1 use max(base, 1).
Prints the row count read and exits non-zero when it is 0 (CLAUDE.md §6).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

NODE_BLOWUP = 10.0


def load(paths):
    rows = {}
    errors = []
    for p in paths:
        for ln in open(p):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if "error" in r:
                errors.append(r)
                continue
            rows[(r["name"], r["arm"])] = r  # last write wins (resume semantics)
    return rows, errors


def fmt_gap(g):
    return "-" if g is None else f"{g:.2g}"


def main(paths):
    rows, errors = load(paths)
    if not rows:
        print("NO ROWS READ")
        return 2
    names = sorted({n for n, _ in rows}, key=lambda s: s.lower())
    arms = []
    for _, a in rows:
        if a not in arms:
            arms.append(a)
    order = ["base_tdef", "base"] + sorted(a for a in arms if a not in ("base_tdef", "base"))
    arms = [a for a in order if a in arms]
    print(f"rows read: {len(rows)}; worker errors: {len(errors)}; instances: {len(names)}; "
          f"arms: {arms}")
    for e in errors:
        print("  ERROR", e["name"], e["arm"], e.get("error"), (e.get("stderr_tail") or "")[-200:])
    incorrect = [(k, r["incorrect"]) for k, r in rows.items() if r["incorrect"]]
    print(f"incorrect rows: {len(incorrect)}")
    for k, why in incorrect:
        print("  INCORRECT", k, why)

    # per-instance matrix
    crit = defaultdict(list)       # instance -> [(arm, reason)]
    blow = defaultdict(list)
    for n in names:
        base = rows.get((n, "base"))
        if base is None:
            print(f"\n== {n}: no base row yet")
            continue
        bn = max(base["nodes"], 1)
        print(f"\n== {n}  (n={base['n']} m={base['m']} int={base['n_int']}; base: "
              f"{base['status']} nodes={base['nodes']} wall={base['wall']:.1f}s "
              f"root {base['log']['root_lp_bound']} -> {base['log']['root_end_bound']} "
              f"cuts={base['log']['root_end_cuts']} restarts={base['log']['restarts']} "
              f"src={''.join(s['src'] for s in base['log']['sol_sources'])})")
        print(f"   {'arm':20s} {'status':10s} {'cert':4s} {'nodes':>8s} {'x base':>7s} {'wall':>6s} "
              f"{'pgap':>8s} {'dgap':>8s} {'root_end':>14s} {'cuts':>5s} {'rst':>3s} {'SBit':>7s} {'sepIt':>7s} {'heurIt':>7s}  srcs")
        for a in arms:
            r = rows.get((n, a))
            if r is None:
                continue
            rep = r["log"]["report"]
            ratio = r["nodes"] / bn
            flag = ""
            if a not in ("base", "base_tdef") and not a.startswith("root_only"):
                if not r["certified"]:
                    crit[n].append((a, "fails"))
                    flag = " <-- FAILS"
                elif ratio >= NODE_BLOWUP:
                    blow[n].append((a, f"{ratio:.0f}x nodes"))
                    flag = f" <-- {ratio:.0f}x nodes"
            print(f"   {a:20s} {r['status'][:10]:10s} {'Y' if r['certified'] else 'n':4s} "
                  f"{r['nodes']:8d} {ratio:7.1f} {r['wall']:6.1f} {fmt_gap(r['primal_gap']):>8s} "
                  f"{fmt_gap(r['dual_gap']):>8s} {str(r['log']['root_end_bound'])[:14]:>14s} "
                  f"{str(r['log']['root_end_cuts']):>5s} {r['log']['restarts']:3d} "
                  f"{str(rep.get('iters_strong_br', '')):>7s} {str(rep.get('iters_separation', '')):>7s} "
                  f"{str(rep.get('iters_heuristics', '')):>7s}  "
                  f"{''.join(s['src'] for s in r['log']['sol_sources'])}{flag}")

    # aggregate ranking
    print("\n==== Critical components (ablation FAILS to certify at 20 s) ====")
    cnt = defaultdict(list)
    for n, lst in crit.items():
        for a, _ in lst:
            cnt[a].append(n)
    for a, lst in sorted(cnt.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a:20s} {len(lst):2d}  {', '.join(sorted(lst))}")
    print("\n==== Node blow-up >= 10x (still certified) ====")
    cnt2 = defaultdict(list)
    for n, lst in blow.items():
        for a, why in lst:
            cnt2[a].append(f"{n}({why})")
    for a, lst in sorted(cnt2.items(), key=lambda kv: -len(kv[1])):
        print(f"  {a:20s} {len(lst):2d}  {', '.join(sorted(lst))}")
    print("\n==== Per-instance minimal critical set (fails | blow-ups) ====")
    for n in names:
        f = [a for a, _ in crit.get(n, [])]
        b = [f"{a}:{w}" for a, w in blow.get(n, [])]
        print(f"  {n:20s} FAIL={f or '-'}  BLOWUP={b or '-'}")
    # root-only decomposition
    ro_arms = [a for a in arms if a.startswith("root_only")]
    if ro_arms:
        print("\n==== Root-only (mip_max_nodes=1) bound / incumbent decomposition: dual gap % | primal gap % ====")
        print(f"   {'instance':20s} " + " ".join(f"{a.replace('root_only', 'ro'):>22s}" for a in ro_arms))
        for n in names:
            cells = []
            for a in ro_arms:
                r = rows.get((n, a))
                if r is None:
                    cells.append(f"{'-':>22s}")
                    continue
                dg = "-" if r["dual_gap"] is None else f"{100 * r['dual_gap']:.2f}"
                pg = "-" if r["primal_gap"] is None else f"{100 * r['primal_gap']:.2f}"
                cells.append(f"{dg + ' | ' + pg + ' ' + str(r['log']['restarts']) + 'r':>22s}")
            print(f"   {n:20s} " + " ".join(cells))
    base_tdef_diff = []
    for n in names:
        a, b = rows.get((n, "base_tdef")), rows.get((n, "base"))
        if a and b and (a["certified"] != b["certified"] or a["nodes"] != b["nodes"]):
            base_tdef_diff.append((n, a["certified"], a["nodes"], b["certified"], b["nodes"]))
    print(f"\nthreads=default vs threads=1 differ on {len(base_tdef_diff)} instances: {base_tdef_diff}")
    print(f"\nsummary rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
