"""Regression tests for issue #932: the POUNCE ``PyNlProblem`` must never
cross threads.

``PyNlProblem`` is a pyo3 ``unsendable`` pyclass: pyo3 panics on any access
from a thread other than the creating one ("unsendable, but sent to another
thread" — the exact message in #932) and on a cross-thread dealloc
("unsendable, but is being dropped on another thread"). Before the fix,
``get_native_base`` cached the base on the Model, which broke the contract two
ways; each test below pins one of them and fails on the pre-#932 cache:

* **Access**: a second thread calling ``get_native_base`` on the same model
  received the first thread's ``PyNlProblem``, and using it panicked. Now each
  thread builds and uses its own base.
* **Drop**: a discarded Model is *cyclic garbage* (its expression DAG holds
  reference cycles), so the model-attribute cache put the unsendable object
  into a cycle whose collection — and hence the object's dealloc — ran on
  whichever thread triggered the next cyclic GC (a pytest-xdist service
  thread, a POUNCE callback worker, ...). Now the base is not reachable from
  the model at all, so collecting the model's cycle on a foreign thread frees
  no unsendable object.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import weakref

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model

pytestmark = [pytest.mark.smoke, pytest.mark.requires_pounce]


def _convex_nlp():
    m = Model("native932")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 1.0)
    m.minimize((x - 0.2) ** 2 + (y - 0.1) ** 2)
    return m


def _base_or_skip(ev):
    from discopt.solvers.nlp_native import get_native_base

    nb = get_native_base(ev)
    if nb is None:
        pytest.skip("POUNCE-native base unavailable in this environment")
    return nb


class _UnraisableRecorder:
    """Record unraisable exceptions (pyo3 surfaces dealloc panics there)."""

    def __init__(self):
        self.seen: list[str] = []
        self._prev = None

    def __enter__(self):
        self._prev = sys.unraisablehook
        sys.unraisablehook = self._hook
        return self

    def _hook(self, unraisable):
        self.seen.append(f"{unraisable.exc_type.__name__}: {unraisable.exc_value}")

    def __exit__(self, *exc):
        sys.unraisablehook = self._prev
        return False


def test_foreign_thread_gets_its_own_base_and_can_solve():
    """A second thread must never receive (or touch) the first thread's base.

    Pre-#932 the model-level cache returned the main thread's PyNlProblem to
    the worker, and ``solve_node_native`` died with pyo3's "unsendable, but
    sent to another thread" panic. Now the worker builds its own base and the
    node solve succeeds.
    """
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_native import get_native_base, solve_node_native

    m = _convex_nlp()
    ev = NLPEvaluator(m)
    nb_main = _base_or_skip(ev)
    # Same thread, same model: the cache still returns the same object.
    assert get_native_base(ev) is nb_main

    out: dict = {}

    def worker():
        try:
            nb = get_native_base(ev)
            out["distinct"] = nb is not nb_main
            res = solve_node_native(
                nb,
                np.array([0.0, 0.0]),
                np.array([-2.0, -2.0]),
                np.array([2.0, 2.0]),
                {"max_iter": 200},
            )
            out["status"] = res.status
            out["objective"] = res.objective
        except BaseException as e:  # noqa: BLE001 - the pre-fix failure is a PanicException
            out["error"] = repr(e)

    t = threading.Thread(target=worker, name="native-base-worker")
    t.start()
    t.join()

    assert "error" not in out, f"foreign-thread use of the native base failed: {out['error']}"
    assert out["distinct"], "cache handed one thread's unsendable PyNlProblem to another"
    assert out["status"] == SolveStatus.OPTIMAL
    # KKT for min (x-.2)^2+(y-.1)^2 s.t. x+y>=1: x*=0.55, y*=0.45.
    assert out["objective"] == pytest.approx(2 * 0.35**2, abs=1e-6)


def test_model_cycle_collected_on_foreign_thread_frees_no_unsendable(capfd):
    """Collecting a discarded model's cycle on a foreign thread must be silent.

    The test first proves the probe fires: the discarded model really is
    cyclic garbage (weakref survives the del) and the foreign-thread
    ``gc.collect`` really collects it (weakref dies). Pre-#932 that same
    collection dealloc'd the cached PyNlProblem on the collecting thread and
    pyo3 panicked ("unsendable, but is being dropped on another thread");
    post-fix the base lives in this thread's cache, not in the model's cycle.
    """
    m = _convex_nlp()
    ev = NLPEvaluator(m)
    _base_or_skip(ev)

    model_ref = weakref.ref(m)
    del m, ev
    gc_ran = threading.Event()

    assert model_ref() is not None, (
        "probe no longer fires: the model was freed by refcount alone, so this "
        "test cannot exercise foreign-thread cyclic GC any more"
    )

    def collector():
        gc.collect()
        gc_ran.set()

    with _UnraisableRecorder() as rec:
        t = threading.Thread(target=collector, name="foreign-gc")
        t.start()
        t.join()

    assert gc_ran.is_set()
    assert model_ref() is None, "foreign-thread gc.collect did not collect the model cycle"
    unsendable = [s for s in rec.seen if "unsendable" in s]
    assert not unsendable, f"unsendable object freed on a foreign thread: {unsendable[0]}"
    err = capfd.readouterr().err
    assert "unsendable" not in err, f"pyo3 unsendable panic on stderr:\n{err}"


def test_stale_entries_evicted_on_owning_thread():
    """Cache housekeeping (dead models, cap overflow) drops bases on the owner.

    Builds more distinct models than the per-thread cap on one thread and
    verifies the evictions happen inline (same thread) without a panic, and
    that the cache still serves the most recent model.
    """
    from discopt.solvers import nlp_native
    from discopt.solvers.nlp_native import get_native_base

    keep = []
    with _UnraisableRecorder() as rec:
        for _ in range(nlp_native._TLS_MAX_ENTRIES + 2):
            m = _convex_nlp()
            ev = NLPEvaluator(m)
            nb = _base_or_skip(ev)
            keep.append((m, ev, nb))
        entries = nlp_native._tls_entries()
        assert len(entries) <= nlp_native._TLS_MAX_ENTRIES
        # The newest model is still served from cache.
        m, ev, nb = keep[-1]
        assert get_native_base(ev) is nb
    assert not [s for s in rec.seen if "unsendable" in s]
