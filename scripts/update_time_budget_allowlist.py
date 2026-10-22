#!/usr/bin/env python
"""Build python/tests/time_budget_allowlist.json from a call-phase junit report.

The fast lane's ``--time-budget`` (python/tests/_time_budget.py) fails any test
whose call phase exceeds the budget unless the test is allowlisted. This writes the
allowlist from a full run of the lane, keeping every test measured above
``budget / RETIRE_FACTOR``: a test not on the list then needs a RETIRE_FACTOR-fold
slowdown to trip the budget, so runner noise cannot fail it.

    pytest python/tests/ <the python-fast marker/ignore flags> -n 4 \\
        -o junit_duration_report=call --junitxml=fast_call.xml
    python scripts/update_time_budget_allowlist.py fast_call.xml --budget 15

Run it to (re)create the list deliberately; day to day the list only shrinks, as
its tests are sped up or marked slow.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _time_budget import ALLOWLIST_PATH, RETIRE_FACTOR  # noqa: E402
from update_test_durations import nodeid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("junitxml", type=Path)
    ap.add_argument("--budget", type=float, required=True)
    args = ap.parse_args()

    root = ET.parse(args.junitxml).getroot()
    threshold = args.budget / RETIRE_FACTOR
    seen = 0
    unresolved: list[str] = []
    allow: dict[str, float] = {}
    for tc in root.iter("testcase"):
        if not tc.get("classname"):
            continue  # a module-level skip
        seen += 1
        nid = nodeid(tc.get("classname", ""), tc.get("name", ""))
        if nid is None:
            unresolved.append(f"{tc.get('classname')}::{tc.get('name')}")
            continue
        took = float(tc.get("time", 0.0))
        if took > threshold:
            allow[nid] = round(took, 2)
    if seen == 0:
        print("no <testcase> elements in the report", file=sys.stderr)
        return 1
    if unresolved:
        print(
            f"{len(unresolved)} testcases did not map to a file, e.g. {unresolved[:3]}",
            file=sys.stderr,
        )
        return 1
    ALLOWLIST_PATH.write_text(json.dumps(dict(sorted(allow.items())), indent=0) + "\n")
    print(
        f"{seen} tests read; {len(allow)} over {threshold:g} s (budget {args.budget:g} s / "
        f"{RETIRE_FACTOR}) written to {ALLOWLIST_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
