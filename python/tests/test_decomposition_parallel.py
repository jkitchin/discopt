"""Tests for the Decomposition Advisor parallel-execution layer (Phase 6).

Covers the CommunicationLayer backends (sequential + thread pool), the
SchedulingGraph (big-first order, cost model), and DecomposedModel.map_subproblems
(deterministic block-order reduce, straggler-avoiding execution order).
"""

import threading

import discopt.modeling as dm
import pytest
from discopt.decomposition import (
    SchedulingGraph,
    SequentialComm,
    ThreadPoolComm,
    analyze_decomposition,
)
from discopt.decomposition.parallel import build_schedule, select_backend
from discopt.decomposition.parallel.schedule import ScheduledTask

# ── fixtures ───────────────────────────────────────────────────


def _independent_blocks_model():
    m = dm.Model("indep")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    u = m.continuous("u", lb=0, ub=1)
    v = m.continuous("v", lb=0, ub=1)
    m.subject_to(x + y <= 1)
    m.subject_to(u + v <= 1)
    m.minimize(x + y + u + v)
    return m


def _benders_model():
    m = dm.Model("benders")
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.subject_to(x <= 5 * z)
    m.subject_to(y <= 5 * z)
    m.minimize(x + y - z)
    return m


# ── communication backends ─────────────────────────────────────


def test_sequential_map_preserves_order():
    assert SequentialComm().map([1, 2, 3], lambda x: x * x) == [1, 4, 9]


def test_threadpool_map_preserves_order():
    assert ThreadPoolComm(max_workers=4).map([1, 2, 3, 4], lambda x: x + 10) == [11, 12, 13, 14]


def test_threadpool_actually_uses_threads():
    seen = set()
    lock = threading.Lock()

    def worker(x):
        with lock:
            seen.add(threading.get_ident())
        return x

    ThreadPoolComm(max_workers=4).map(list(range(8)), worker)
    # tasks ran on pool worker threads, never on the calling (main) thread
    assert seen
    assert threading.get_ident() not in seen


def test_backends_agree_on_pure_function():
    def fn(x):
        return x * 3 - 1

    items = list(range(10))
    assert SequentialComm().map(items, fn) == ThreadPoolComm().map(items, fn)


def test_select_backend_names_and_passthrough():
    assert select_backend("sequential").name == "sequential"
    assert select_backend("threads").name == "threads"
    inst = SequentialComm()
    assert select_backend(inst) is inst
    with pytest.raises(ValueError):
        select_backend("mpi")  # not shipped yet


def test_empty_map_returns_empty():
    assert SequentialComm().map([], lambda x: x) == []
    assert ThreadPoolComm().map([], lambda x: x) == []


# ── scheduling graph ───────────────────────────────────────────


def test_schedule_execution_order_is_big_first():
    g = SchedulingGraph(
        tasks=(
            ScheduledTask(0, 1.0),
            ScheduledTask(1, 5.0),
            ScheduledTask(2, 3.0),
        )
    )
    assert g.execution_order() == [1, 2, 0]  # descending cost


def test_schedule_order_is_deterministic_on_ties():
    g = SchedulingGraph(tasks=(ScheduledTask(2, 1.0), ScheduledTask(0, 1.0), ScheduledTask(1, 1.0)))
    assert g.execution_order() == [0, 1, 2]  # ties broken by block id


def test_schedule_cost_model():
    g = SchedulingGraph(tasks=(ScheduledTask(0, 2.0), ScheduledTask(1, 8.0)))
    assert g.total_cost() == 10.0
    assert g.critical_path_cost() == 8.0
    assert g.ideal_speedup() == pytest.approx(1.25)


def test_build_schedule_from_decomposition():
    dcmp = analyze_decomposition(_independent_blocks_model()).decompose()
    g = build_schedule(dcmp)
    assert g.num_blocks == 2
    assert not g.has_master  # independent blocks: no master
    assert "SchedulingGraph" in g.summary()


def test_build_schedule_benders_has_master():
    dcmp = analyze_decomposition(_benders_model()).decompose()
    assert build_schedule(dcmp).has_master


# ── DecomposedModel.map_subproblems ────────────────────────────


def test_map_subproblems_reduces_in_block_order():
    dcmp = analyze_decomposition(_independent_blocks_model()).decompose()
    results = dcmp.map_subproblems(lambda sp: sp.block_id, backend="sequential")
    # results indexed by block order regardless of execution order
    assert results == [0, 1]


def test_map_subproblems_executes_big_first():
    # build a decomposition with unequal block sizes and record execution order
    dcmp = analyze_decomposition(_benders_model()).decompose()
    # both recourse blocks have size 1 here; craft explicit uneven sizes instead
    from discopt.decomposition.ir.models import SubproblemModel

    dcmp.subproblems = [
        SubproblemModel(0, ("a",)),
        SubproblemModel(1, ("b", "c", "d")),  # bigger → should run first
        SubproblemModel(2, ("e", "f")),
    ]
    order = []
    dcmp.map_subproblems(lambda sp: order.append(sp.block_id), backend="sequential")
    assert order == [1, 2, 0]  # descending size


def test_map_subproblems_threads_and_sequential_agree():
    dcmp = analyze_decomposition(_independent_blocks_model()).decompose()
    seq = dcmp.map_subproblems(lambda sp: sp.size, backend="sequential")
    thr = dcmp.map_subproblems(lambda sp: sp.size, backend="threads")
    assert seq == thr


def test_decomposed_model_schedule_method():
    dcmp = analyze_decomposition(_benders_model()).decompose()
    g = dcmp.schedule()
    assert g.num_blocks == dcmp.num_blocks
