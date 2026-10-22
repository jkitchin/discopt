"""The CI shard selector partitions the selection exactly (python/tests/_sharding.py).

A shard that silently dropped or duplicated tests would read as a green run
while executing less than the lane; these pin the partition itself. The CI
aggregator jobs check the same property on the real run from --shard-report.
"""

import json
import random

import _sharding
import pytest

pytest_plugins = ["pytester"]


def _ids(n):
    return [f"python/tests/test_m{i % 37}.py::test_{i}[p{i % 5}]" for i in range(n)]


@pytest.mark.parametrize("num_shards", [1, 2, 3, 4, 7])
def test_every_test_lands_in_exactly_one_shard(num_shards):
    ids = _ids(1000)
    rng = random.Random(0)
    durations = {i: rng.choice([0.01, 0.3, 2.0, 40.0]) for i in ids[::3]}
    shards = _sharding.assign_shards(ids, num_shards, durations)
    assert len(shards) == len(ids)
    assert set(shards) == set(range(num_shards))


def test_assignment_does_not_depend_on_collection_order():
    ids = _ids(500)
    durations = {i: 1.0 + (hash(i) % 7) for i in ids[::2]}
    a = dict(zip(ids, _sharding.assign_shards(ids, 4, durations)))
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    b = dict(zip(shuffled, _sharding.assign_shards(shuffled, 4, durations)))
    assert a == b


def test_heavy_tests_are_spread_not_stacked():
    ids = _ids(400)
    heavy = ids[:8]
    durations = {i: 60.0 for i in heavy}
    shards = dict(zip(ids, _sharding.assign_shards(ids, 4, durations)))
    loads = [
        sum(durations.get(i, _sharding.DEFAULT_DURATION_S) for i in ids if shards[i] == s)
        for s in range(4)
    ]
    assert sorted(shards[i] for i in heavy) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert max(loads) - min(loads) <= 60.0


def test_rejects_zero_shards():
    with pytest.raises(ValueError):
        _sharding.assign_shards(["a"], 0, {})


def test_shards_partition_a_real_session(pytester, monkeypatch):
    """End to end through pytest: the shards' selections and reports add up."""
    # The in-process run must not look like an xdist worker other than gw0, or
    # it would (correctly) leave the report to gw0 and write none.
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    pytester.makepyfile(
        **{f"test_s{k}": "\n".join(f"def test_{j}(): pass" for j in range(9)) for k in range(5)}
    )
    pytester.makeconftest(
        "import _sharding\n"
        "def pytest_addoption(parser):\n"
        "    _sharding.addoption(parser)\n"
        "def pytest_configure(config):\n"
        "    config.pluginmanager.register(_sharding.ShardPlugin(), 'discopt-shard')\n"
    )
    ran, reports = [], []
    for sid in range(3):
        rep = pytester.path / f"rep{sid}.json"
        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            "-v",
            "-k",
            "not test_8",
            "--num-shards",
            "3",
            "--shard-id",
            str(sid),
            "--shard-report",
            str(rep),
        )
        result.assert_outcomes(passed=json.loads(rep.read_text())["mine"])
        ran += [ln.split(" ")[0] for ln in result.outlines if " PASSED" in ln]
        reports.append(json.loads(rep.read_text()))
    assert len(ran) == len(set(ran)) == 5 * 8  # -k ran first: test_8 is not counted
    assert {r["total"] for r in reports} == {40}
    assert sum(r["mine"] for r in reports) == 40


def test_shard_id_without_count_is_an_error(pytester):
    pytester.makepyfile(test_a="def test_a(): pass")
    pytester.makeconftest(
        "import _sharding\n"
        "def pytest_addoption(parser):\n"
        "    _sharding.addoption(parser)\n"
        "def pytest_configure(config):\n"
        "    config.pluginmanager.register(_sharding.ShardPlugin(), 'discopt-shard')\n"
    )
    result = pytester.runpytest("-p", "no:cacheprovider", "--shard-id", "0")
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*--num-shards and --shard-id must be given together*"])
