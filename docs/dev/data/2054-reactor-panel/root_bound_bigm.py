"""#2054: root relaxation bound under gdp_method="big-m", DISCOPT_GDP_BIGM_SUBDIVISION OFF vs ON.

A time-limited tree comparison mixes the M with tree-size noise; the root bound is
deterministic and measures what a smaller M buys directly. Every root bound is
checked against reactor_panel's independent verified-feasible oracle (a root bound
above it is a false bound). Prints per-model rows, a summary and the executed-check
count; exits non-zero on a violation or when nothing was checked.

Usage: python -u root_bound_bigm.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from reactor_panel import TOL_ABS, TOL_REL, build, family, oracle


def main():
    import discopt._relax.gdp_reformulate as G

    assert "_subdivided_body_bounds" in open(G.__file__).read(), G.__file__
    os.environ["DISCOPT_OA_CONVEXITY_SUBDIVISION"] = "0"
    checks, bad = 0, []
    gains = []
    for spec in family(False):
        opt, _ = oracle(spec)
        rb = {}
        for flag in ("0", "1"):
            os.environ["DISCOPT_GDP_BIGM_SUBDIVISION"] = flag
            r = build(spec).solve(gdp_method="big-m", max_nodes=1, time_limit=60)
            rb[flag] = r.root_bound
            if r.root_bound is not None:
                checks += 1
                if r.root_bound > opt + TOL_ABS + TOL_REL * max(1.0, abs(opt)):
                    bad.append(f"{spec[0]} flag={flag}: root bound {r.root_bound} > feasible {opt}")
        a, b = rb["0"], rb["1"]
        gap_a = None if a is None else opt - a
        gap_b = None if b is None else opt - b
        closed = None
        if gap_a is not None and gap_b is not None and gap_a > 1e-9:
            closed = (gap_a - gap_b) / gap_a
            gains.append(closed)
        print(f"{spec[0]}: oracle {opt:.6f} root OFF {a} ON {b} gap closed {closed}", flush=True)
    g = np.array(gains)
    print(
        f"models {len(g)}: root gap closed median {np.median(g):.3f}, mean {g.mean():.3f}, "
        f"improved {(g > 1e-6).sum()}, worse {(g < -1e-6).sum()}, same {(abs(g) <= 1e-6).sum()}"
    )
    print("root bounds checked against the oracle:", checks, "violations:", len(bad))
    for line in bad:
        print("  FAIL", line)
    if checks == 0 or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
