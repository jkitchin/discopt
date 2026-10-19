"""Serialization ratio budgets on five fixed in-repo instances (WP-7, D13).

Plan: ``docs/dev/serialization-completion-plan.md`` WP-7. The budgets are ratios
of walls measured in the same process, so they hold on any machine; the
constants and the measurement live in ``scripts/serialize_perf_ladder.py``,
which also runs the full ladder. The instances are the five largest of the
in-repo ``.nl`` floor by ``dumps`` wall -- each denominator well above the
ladder's ``MIN_DENOM_S`` noise floor, so every ratio here is judged.

``slow``: about 3 s of best-of-five timing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
DATA = Path(__file__).parent / "data"
_spec = importlib.util.spec_from_file_location(
    "serialize_perf_ladder", REPO / "scripts" / "serialize_perf_ladder.py"
)
ladder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ladder
_spec.loader.exec_module(ladder)

INSTANCES = (
    DATA / "minlplib_nl" / "hda.nl",
    DATA / "minlplib_nl" / "heatexch_gen3.nl",
    DATA / "minlplib" / "carton7.nl",
    DATA / "minlplib_nl" / "casctanks.nl",
    DATA / "minlplib_nl" / "contvar.nl",
)
REPEATS = 5

_KEYS = ("rt_over_from_nl", "gz_over_nl", "pickle_over_dumps")


@pytest.fixture(scope="module")
def rows():
    return [ladder.measure_instance(p, REPEATS) for p in INSTANCES]


def _judge(rows, key, budget):
    over = []
    for row in rows:
        value = row[key]
        assert value is not None, (row["name"], key, "below the noise floor: pick another")
        if value > budget:
            over.append((row["name"], round(value, 3)))
    return over


def test_round_trip_within_ratio_budget(rows):
    """``loads(dumps(m))`` within 5x ``from_nl``; ``pickle.dumps`` within 1.25x ``dumps``."""
    assert not _judge(rows, "rt_over_from_nl", ladder.RT_OVER_FROM_NL)
    assert not _judge(rows, "pickle_over_dumps", ladder.PICKLE_OVER_DUMPS)


def test_document_size_within_ratio_budget(rows):
    """The gzipped document within 3x the ``.nl`` bytes."""
    assert not _judge(rows, "gz_over_nl", ladder.GZ_OVER_NL)


def test_budget_probe_fired(rows):
    """§6: every instance was measured, and every ratio of every instance is one
    the budget tests judge (none fell below a noise floor and was skipped).

    Recomputed from the rows rather than counted by the tests above, so it holds
    when ``-n`` puts them in different workers."""
    assert [r["name"] for r in rows] == [p.stem for p in INSTANCES]
    judged = sum(r[k] is not None for r in rows for k in _KEYS)
    assert judged == len(_KEYS) * len(INSTANCES), judged
