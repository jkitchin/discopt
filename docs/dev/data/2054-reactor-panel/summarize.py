"""Summarise reactor_panel.py JSONL output per flag, per method (#2054).

Compares arm 10 vs 00 (the OA subdivision flag). The recorded 9b6a3c3 run also has
the retired big-M arms (01, 11); they are reported when present.
Bar 1 (cert-clean) is read from the panel's own failure list; this reports bar 2:
certified counts, certification regressions, bound movement on the instances
neither arm certifies, and node/iteration/wall totals over the instances both arms
certify.

Usage: python summarize.py RUN.jsonl [RUN.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def main():
    rows = [json.loads(line) for p in sys.argv[1:] for line in open(p) if line.strip()]
    assert rows, "no rows"
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], r["method"])][r["sub"] + r["bigm"]] = r
    n_cmp = 0
    for method in ("loa", "big-m"):
        for label, off, on in (
            ("OA subdivision", "00", "10"),
            ("big-M subdivision", "00", "01"),
            ("both", "00", "11"),
        ):
            keys = [k for k in by if k[1] == method and off in by[k] and on in by[k]]
            cert = {a: 0 for a in (off, on)}
            gained, lost = [], []
            both_cert = []
            open_bound = []
            for k in keys:
                a, b = by[k][off], by[k][on]
                ca, cb = a.get("status") == "optimal", b.get("status") == "optimal"
                cert[off] += ca
                cert[on] += cb
                n_cmp += 1
                if cb and not ca:
                    gained.append(k[0])
                if ca and not cb:
                    lost.append(k[0])
                if ca and cb:
                    both_cert.append((a, b))
                if not ca and not cb and a.get("bound") is not None and b.get("bound") is not None:
                    open_bound.append(b["bound"] - a["bound"])
            neg = [k for k in keys if k[0].startswith("neg")]

            def tot(rs, f):
                return sum((r.get(f) or 0) for r in rs)

            print(f"[{method}] {label}: {on} vs {off} over {len(keys)} models ({len(neg)} neg)")
            print(f"    certified {cert[off]} -> {cert[on]}   gained {len(gained)} {gained}")
            print(f"    LOST (certification regressions) {len(lost)} {lost}")
            if both_cert:
                A = [p[0] for p in both_cert]
                B = [p[1] for p in both_cert]
                print(
                    f"    both certify {len(both_cert)}: nodes {tot(A, 'nodes')} -> {tot(B, 'nodes')}, "
                    f"iterations {tot(A, 'iterations')} -> {tot(B, 'iterations')}, "
                    f"wall {tot(A, 'wall'):.1f} -> {tot(B, 'wall'):.1f} s"
                )
            if open_bound:
                d = np.array(open_bound)
                print(
                    f"    neither certifies {len(d)}: bound tighter {(d > 1e-9).sum()}, "
                    f"looser {(d < -1e-9).sum()}, same {(abs(d) <= 1e-9).sum()}"
                )
            wall_all_a = sum(by[k][off]["wall"] for k in keys)
            wall_all_b = sum(by[k][on]["wall"] for k in keys)
            print(f"    total wall all models {wall_all_a:.1f} -> {wall_all_b:.1f} s")
    print("comparisons executed:", n_cmp)
    if n_cmp == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
