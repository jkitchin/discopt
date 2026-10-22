"""Duration-balanced test sharding for the CI matrix (``--num-shards``/``--shard-id``).

The PR-fast and coverage lanes ran 18k tests on one 4-vCPU runner: 37 min for
python-fast and 65 min for python-coverage (CI run 39688620069, which then
failed). A profile of that lane (4-core box, ``-n 4``) put 72% of the test time
in the 465 tests that take >= 2 s, so splitting by test *count* would leave one
shard holding most of the heavy tail. Instead each test is weighted by its
recorded duration (``test_durations.json``, refreshed by
``scripts/update_test_durations.py``) and assigned greedily, longest first, to
the least-loaded shard (LPT). A test with no recorded duration gets
``DEFAULT_DURATION_S``, so a new test is still assigned deterministically.

Every shard collects the same items, so every shard computes the same
partition: each selected test runs in exactly one shard. ``--shard-report``
writes the shard's counts so the CI aggregator can check that the shards add
up to the whole selection (CLAUDE.md §6) instead of trusting it.
"""

from __future__ import annotations

import heapq
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

DURATIONS_PATH = Path(__file__).with_name("test_durations.json")
DEFAULT_DURATION_S = 0.05


def load_durations(path: Path = DURATIONS_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {str(k): float(v) for k, v in data.items()}


def assign_shards(
    nodeids: Sequence[str], num_shards: int, durations: Mapping[str, float]
) -> list[int]:
    """Shard index for each nodeid; deterministic in the *set* of nodeids."""
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    order = sorted(
        range(len(nodeids)),
        key=lambda i: (-durations.get(nodeids[i], DEFAULT_DURATION_S), nodeids[i]),
    )
    heap = [(0.0, s) for s in range(num_shards)]
    out = [-1] * len(nodeids)
    for i in order:
        load, s = heapq.heappop(heap)
        out[i] = s
        heapq.heappush(heap, (load + durations.get(nodeids[i], DEFAULT_DURATION_S), s))
    return out


def addoption(parser) -> None:
    group = parser.getgroup("sharding")
    group.addoption("--num-shards", type=int, default=None, help="split the run into N shards")
    group.addoption("--shard-id", type=int, default=None, help="run shard K (0-based)")
    group.addoption(
        "--shard-report", default=None, help="write this shard's counts as JSON to PATH"
    )


def select_shard(config, items) -> None:
    num = config.getoption("--num-shards")
    sid = config.getoption("--shard-id")
    if num is None and sid is None:
        return
    if num is None or sid is None:
        raise ValueError("--num-shards and --shard-id must be given together")
    if not 0 <= sid < num:
        raise ValueError(f"--shard-id {sid} out of range for --num-shards {num}")
    durations = load_durations()
    shards = assign_shards([it.nodeid for it in items], num, durations)
    keep = [it for it, s in zip(items, shards) if s == sid]
    drop = [it for it, s in zip(items, shards) if s != sid]
    report = config.getoption("--shard-report")
    # Under xdist every worker collects and runs this hook with the same items;
    # one of them (or the lone process without xdist) writes the report.
    if report and os.environ.get("PYTEST_XDIST_WORKER") in (None, "gw0"):
        Path(report).write_text(
            json.dumps(
                {
                    "shard_id": sid,
                    "num_shards": num,
                    "total": len(items),
                    "mine": len(keep),
                    "expected_s": round(
                        sum(durations.get(it.nodeid, DEFAULT_DURATION_S) for it in keep), 1
                    ),
                }
            )
        )
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


class ShardPlugin:
    # trylast: partition only what survives ``-m``/``-k`` deselection, so the
    # balance is computed over the tests that actually run.
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        select_shard(config, items)
