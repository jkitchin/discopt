#!/usr/bin/env python
"""Build python/tests/time_budget_allowlist.json from call-phase timings.

The fast lane's ``--time-budget`` (python/tests/_time_budget.py) fails any test
whose call phase exceeds the budget unless the test is allowlisted. This writes the
allowlist, keeping every test measured above ``budget / RETIRE_FACTOR``: a test not
on the list then needs a RETIRE_FACTOR-fold slowdown to trip the budget, so runner
noise cannot fail it.

Build it from **CI timings**, not a dev machine's: the runners are up to ~3x slower
on some tests (QPLIB_3871: 5.4 s local, 17.6 s on CI), which is more than the margin.
Each fast shard uploads ``fast-junit-K.xml`` (call-phase times) in its
``fast-shard-report-K`` artifact and prints every test >= 5 s via ``--durations``;
either form is accepted, several inputs are merged, and a test's largest time wins::

    python scripts/update_time_budget_allowlist.py fast-junit-*.xml --budget 15
    python scripts/update_time_budget_allowlist.py shard*.log --budget 15

A ``.xml`` input is a junit report written with ``-o junit_duration_report=call``;
anything else is read as pytest output, from its ``--durations`` lines
(``12.34s call     path::test``; setup/teardown lines are ignored). A log lists only
the tests above its ``--durations-min``, which must be below the threshold.

Run it to (re)create the list deliberately; day to day the list only shrinks, as
its tests are sped up or marked slow.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python" / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _time_budget import ALLOWLIST_PATH, RETIRE_FACTOR  # noqa: E402
from update_test_durations import nodeid  # noqa: E402

# A GitHub Actions log prefixes each line with a timestamp; allow anything before.
_DURATION_LINE = re.compile(r"(\d+(?:\.\d+)?)s\s+call\s+(\S+::\S.*?)\s*$")


def read_junit(path: Path) -> tuple[dict[str, float], list[str]]:
    times: dict[str, float] = {}
    unresolved: list[str] = []
    for tc in ET.parse(path).getroot().iter("testcase"):
        if not tc.get("classname"):
            continue  # a module-level skip
        nid = nodeid(tc.get("classname", ""), tc.get("name", ""))
        if nid is None:
            unresolved.append(f"{tc.get('classname')}::{tc.get('name')}")
            continue
        times[nid] = float(tc.get("time", 0.0))
    return times, unresolved


def unmask(nid: str) -> str:
    """Recover a node id GitHub's log masking cut short (``test_hi***``).

    Actions replaces any substring equal to a secret with ``***``, so a test name
    can arrive truncated. Accept it only when exactly one test function in the file
    starts with the visible prefix and nothing (no parameter id) follows the mask;
    otherwise refuse, since a guessed key would silently not match.
    """
    path, _, name = nid.partition("::")
    prefix, _, rest = name.partition("***")
    found = re.findall(rf"^def ({re.escape(prefix)}\w*)\(", (ROOT / path).read_text(), re.M)
    if rest or len(found) != 1:
        raise SystemExit(
            f"masked node id {nid!r} cannot be recovered ({len(found)} candidates); "
            "build from the fast-junit-*.xml artifacts instead"
        )
    return f"{path}::{found[0]}"


def read_durations_log(path: Path) -> dict[str, float]:
    times: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = _DURATION_LINE.search(line)
        if m:
            nid = m.group(2)
            times[unmask(nid) if "***" in nid else nid] = float(m.group(1))
    return times


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--budget", type=float, required=True)
    args = ap.parse_args()

    threshold = args.budget / RETIRE_FACTOR
    merged: dict[str, float] = {}
    for path in args.reports:
        if path.suffix == ".xml":
            times, unresolved = read_junit(path)
            if unresolved:
                print(
                    f"{path}: {len(unresolved)} testcases did not map to a file, "
                    f"e.g. {unresolved[:3]}",
                    file=sys.stderr,
                )
                return 1
        else:
            times = read_durations_log(path)
        if not times:
            print(f"{path}: no test timings found", file=sys.stderr)
            return 1
        print(f"{path}: {len(times)} timings")
        for nid, took in times.items():
            merged[nid] = max(took, merged.get(nid, 0.0))

    allow = {nid: round(took, 2) for nid, took in merged.items() if took > threshold}
    ALLOWLIST_PATH.write_text(json.dumps(dict(sorted(allow.items())), indent=0) + "\n")
    print(
        f"{len(merged)} tests read; {len(allow)} over {threshold:g} s (budget {args.budget:g} s"
        f" / {RETIRE_FACTOR}) written to {ALLOWLIST_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
