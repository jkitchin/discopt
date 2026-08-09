"""Re-score the saved #966 panel artifacts against the FULL MINLPLib oracle.

The panel's own soundness check used ``python/tests/_optima.py``, which covers
1 of its 19 instances, and swallowed the lookup failure in a bare ``except``
(CLAUDE.md §7). This re-reads the saved cells -- no re-run needed -- and scores
every arm's bound against ``minlplib.solu``.

Prints the number of bound-vs-oracle comparisons actually executed and exits
non-zero if that is zero (§6). Uncovered instances are reported, never counted
as clean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from minlplib_solu import load, primal_ceiling  # noqa: E402

TOL = 1e-4
ARMS = ("base", "seam", "cand")

table = load()
compared = 0
violations, uncovered, closest = [], set(), []

for rep in (1, 2, 3):
    art = Path(f"discopt_benchmarks/results/issue966_coupled_binding20_rep{rep}.json")
    cells = json.loads(art.read_text())["cells"]
    for c in cells:
        name = c["instance"]
        ceil_ = primal_ceiling(name, table)
        if ceil_ is None:
            uncovered.add(name)
            continue
        for arm in ARMS:
            rec = c[arm]
            b = rec["bound"]
            if b is None:
                continue
            compared += 1
            sense = rec["sense"]
            # For min: a valid dual bound is <= the optimum <= the best primal.
            slack = (ceil_ - b) if sense == "min" else (b - ceil_)
            closest.append((slack, f"rep{rep}:{name}:{arm}", b, ceil_))
            if slack < -TOL * max(1.0, abs(ceil_)):
                violations.append(
                    {
                        "rep": rep,
                        "instance": name,
                        "arm": arm,
                        "sense": sense,
                        "bound": b,
                        "oracle_ceiling": ceil_,
                        "excess": -slack,
                    }
                )

closest.sort(key=lambda t: t[0])
print(
    json.dumps(
        {
            "bound_vs_oracle_comparisons": compared,
            "instances_uncovered_by_solu": sorted(uncovered),
            "violations": violations,
            "tightest_5_margins": [
                {"cell": c[1], "bound": c[2], "oracle_ceiling": c[3], "margin": c[0]}
                for c in closest[:5]
            ],
        },
        indent=2,
    )
)
print(f"ORACLE_COMPARISONS_EXECUTED={compared}")
print(f"ORACLE_VIOLATIONS={len(violations)}")
raise SystemExit(0 if compared else 1)
