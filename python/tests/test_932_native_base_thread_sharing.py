"""Issue #932: discopt shares one POUNCE ``NlProblem`` across threads.

#932 reported a pyo3 panic — ``_pounce::nl_problem::PyNlProblem is unsendable,
but sent to another thread`` — surfacing during result marshaling, and asked for
one of two outcomes: keep the object on one thread, or, "if the type is genuinely
safe to send, establish that and drop ``unsendable``".

POUNCE took the second route. pounce#477 ("Let one NlProblem serve a whole worker
pool") removed the marker after establishing that ``NlTnlp`` is ``Send`` and that
every ``&self`` method evaluates under the GIL, citing the same reasoning #932
did: ``PanicException`` derives from ``BaseException`` and so slips past every
``except Exception`` in a host, and the *drop* path tripped even for code that
never used an instance cross-thread.

So there is no discopt-side thread-affinity bug to fix, and these tests are not
fail-before/pass-after regression tests for one. They pin the *guarantee* discopt
now depends on — ``get_native_base`` caches the base on the model and hands it to
whichever thread solves a node — and they pin the guard that disables the native
path against a POUNCE where the guarantee does not hold. Against such a build the
first test fails loudly here rather than aborting a solve in the field.

Each test proves its probe fired (CLAUDE.md §6): the cross-thread tests assert the
worker really ran on a different thread, and the GC test asserts the model really
was cyclic garbage really collected by the foreign thread.
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
from discopt._relax.nlp_evaluator import NLPEvaluator
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


def _run_on_worker(fn):
    """Run ``fn`` on a second thread; return (result_dict, worker_tid)."""
    out: dict = {}

    def worker():
        out["tid"] = threading.get_ident()
        try:
            fn(out)
        except BaseException as exc:  # noqa: BLE001 - a pyo3 panic is BaseException
            out["error"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=worker, name="native-base-worker")
    t.start()
    t.join()
    return out, out.get("tid")


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


def test_one_base_serves_a_second_thread():
    """The cached base is shared across threads and works there.

    This is the guarantee pounce#477 provides and discopt's model-level cache
    depends on. Against a POUNCE that still marks ``NlProblem`` unsendable, the
    worker's first access raises ``PanicException`` and this test fails.
    """
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_native import get_native_base, solve_node_native

    m = _convex_nlp()
    ev = NLPEvaluator(m)
    nb_main = _base_or_skip(ev)
    main_tid = threading.get_ident()

    def body(out):
        nb = get_native_base(ev)
        out["shared"] = nb is nb_main
        # Evaluate through POUNCE's own AD, then take a bound variant — the two
        # things the node and batch paths respectively do with a base.
        out["objective"] = nb.base.objective(np.asarray(nb.base.x0, dtype=np.float64))
        out["variant"] = nb.variant is not None
        res = solve_node_native(
            nb,
            np.array([0.0, 0.0]),
            np.array([-2.0, -2.0]),
            np.array([2.0, 2.0]),
            {"max_iter": 200},
        )
        out["status"] = res.status
        out["obj"] = res.objective

    out, worker_tid = _run_on_worker(body)

    assert worker_tid is not None and worker_tid != main_tid, (
        "probe did not fire: the body ran on the creating thread, so this test "
        "exercised no cross-thread use at all"
    )
    assert "error" not in out, (
        f"cross-thread use of the native base failed: {out['error']}\n"
        "If this is a pyo3 'unsendable' panic, this POUNCE predates pounce#477 — "
        "upgrade it; discopt shares one NlProblem across threads (discopt#932)."
    )
    assert out["shared"], "the model-level cache should hand every thread the same base"
    assert out["status"] == SolveStatus.OPTIMAL
    # KKT for min (x-.2)^2+(y-.1)^2 s.t. x+y>=1: x*=0.55, y*=0.45.
    assert out["obj"] == pytest.approx(2 * 0.35**2, abs=1e-6)


def test_model_cycle_collected_on_foreign_thread_is_silent():
    """Collecting a discarded model's cycle on a foreign thread must be silent.

    This is the drop crossing #932 actually observed printing mid-run: the base is
    cached on the model, a discarded model is *cyclic* garbage, so the problem's
    dealloc runs on whichever thread triggers the next cyclic GC. pounce#477 makes
    that legal. The test first proves the probe fires — the model really survives
    ``del`` (so it really is cyclic) and the foreign thread's ``gc.collect`` really
    reclaims it.
    """
    m = _convex_nlp()
    ev = NLPEvaluator(m)
    _base_or_skip(ev)

    model_ref = weakref.ref(m)
    del m, ev

    assert model_ref() is not None, (
        "probe no longer fires: the model was freed by refcount alone, so this "
        "test cannot exercise foreign-thread cyclic GC any more"
    )

    main_tid = threading.get_ident()
    with _UnraisableRecorder() as rec:
        out, worker_tid = _run_on_worker(lambda o: o.__setitem__("collected", gc.collect()))

    assert worker_tid is not None and worker_tid != main_tid, "probe did not fire"
    assert "error" not in out, f"foreign-thread gc.collect raised: {out['error']}"
    assert model_ref() is None, "foreign-thread gc.collect did not collect the model cycle"
    unsendable = [s for s in rec.seen if "unsendable" in s]
    assert not unsendable, f"unsendable object freed on a foreign thread: {unsendable[0]}"


def test_capability_probe_reports_support_and_is_measured_once():
    """The guard measures the guarantee on a real second thread, once per process."""
    from discopt.solvers import nlp_native

    m = _convex_nlp()
    ev = NLPEvaluator(m)
    nb = _base_or_skip(ev)

    assert nlp_native._probe_cross_thread_use(nb) is True

    calls = {"n": 0}

    def counting_probe(_nb):
        calls["n"] += 1
        return True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nlp_native, "_CROSS_THREAD_OK", None)
        mp.setattr(nlp_native, "_probe_cross_thread_use", counting_probe)
        assert nlp_native._cross_thread_use_supported(nb) is True
        assert nlp_native._cross_thread_use_supported(nb) is True
    assert calls["n"] == 1, f"capability probed {calls['n']} times; it must be measured once"


def test_native_path_disabled_when_cross_thread_use_is_unsupported():
    """An older POUNCE must disable the native path, not abort a solve.

    A cross-thread ``PanicException`` derives from ``BaseException`` and so would
    slip past every ``except Exception`` in the solver. The guard therefore refuses
    the native path up front; the caller falls back to the JAX bridge.
    """
    from discopt.solvers import nlp_native

    m = _convex_nlp()
    ev = NLPEvaluator(m)
    _base_or_skip(ev)  # skip early if the native path is unavailable for other reasons

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nlp_native, "_CROSS_THREAD_OK", None)
        mp.setattr(nlp_native, "_probe_cross_thread_use", lambda _nb: False)
        fresh = _convex_nlp()
        assert nlp_native.get_native_base(NLPEvaluator(fresh)) is None
        # And the refusal is cached on the model as "unavailable", not re-probed.
        assert fresh._native_nlp_base_cache[0] is False


def test_probe_refuses_to_pass_if_it_did_not_cross_a_thread():
    """The capability probe must never certify support without a real crossing."""
    from discopt.solvers import nlp_native

    m = _convex_nlp()
    ev = NLPEvaluator(m)
    nb = _base_or_skip(ev)

    class _SameThread(threading.Thread):
        """Runs the target inline, so the 'worker' is the calling thread."""

        def start(self):
            self.run()

        def join(self, timeout=None):
            return None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(nlp_native.threading, "Thread", _SameThread)
        with pytest.raises(RuntimeError, match="proves nothing"):
            nlp_native._probe_cross_thread_use(nb)
