#!/usr/bin/env python
"""Refresh python/tests/test_durations.json from a pytest ``--junitxml`` report.

The CI shard selector (python/tests/_sharding.py) balances shards by these
durations. Stale entries only cost balance, never correctness: every test
still runs in exactly one shard. Refresh when one shard runs much longer than
the others, from a full (unsharded) run of the lane:

    pytest python/tests/ <the python-fast marker/ignore flags> -n 4 --junitxml=fast.xml
    python scripts/update_test_durations.py fast.xml

Tests faster than ``--min-seconds`` are omitted; the selector charges them
``DEFAULT_DURATION_S``.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "python" / "tests" / "test_durations.json"


def nodeid(classname: str, name: str) -> str | None:
    """``a.b.test_x.TestY`` + ``test_z`` -> ``a/b/test_x.py::TestY::test_z``."""
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        path = "/".join(parts[:cut]) + ".py"
        if (ROOT / path).is_file():
            return "::".join([path, *parts[cut:], name])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("junitxml", type=Path)
    ap.add_argument("--min-seconds", type=float, default=0.25)
    args = ap.parse_args()

    seen = kept = 0
    unresolved: list[str] = []
    durations: dict[str, float] = {}
    for tc in ET.parse(args.junitxml).getroot().iter("testcase"):
        seen += 1
        if not tc.get("classname"):
            continue  # a module-level skip: no test item, nothing to balance
        nid = nodeid(tc.get("classname", ""), tc.get("name", ""))
        if nid is None:
            unresolved.append(f"{tc.get('classname')}::{tc.get('name')}")
            continue
        t = float(tc.get("time", 0.0))
        if t >= args.min_seconds:
            durations[nid] = round(t, 2)
            kept += 1
    if seen == 0:
        print("no <testcase> elements in the report", file=sys.stderr)
        return 1
    if unresolved:
        print(
            f"{len(unresolved)} testcases did not map to a file, e.g. {unresolved[:3]}",
            file=sys.stderr,
        )
        return 1
    OUT.write_text(json.dumps(dict(sorted(durations.items())), indent=0) + "\n")
    print(f"{seen} testcases read, {kept} >= {args.min_seconds}s written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
