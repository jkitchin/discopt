"""#2289: every native LP/MILP solve is charged to the ``rust`` bucket.

``python_time`` is a *residual* (wall minus the FFI-boundary counters of
:mod:`discopt._timing`), so a native call that no ``charge("rust")`` region
encloses is silently reported as interpreted Python. Before #2289 only one of the
eight ``solve_lp_warm_csc_py`` call sites was charged: OBBT's probe LPs (2.47 s of
a 10 s ``nvs05`` solve, measured with cProfile), the node-LP warm solve, the
standard-form LP engine, the dual-bound refinement LPs and every
``solve_milp_csc_py`` call all fell into ``python_time`` -- the number the phase1
``interop_overhead`` gate reads.

The test spies on native entry points in ``discopt._rust`` (every call site
imports them from the module at call time) and asserts that the time measured
inside them is covered by the solve's own ``rust`` self time
(``rust_time - pounce_time``). The spy's wall is an upper bound on the native
self time it encloses only while no Python callback runs inside the call.
``solve_spatial_tree_py`` and ``solve_milp_lazy_csc_py`` always take callbacks, so
they are left out. ``solve_milp_csc_py`` and ``solve_milp_relu_csc_py`` take optional
``debug_hook`` / ``node_bound_hook`` callbacks (wrapped in
``_timing.as_interpreted`` at every call site). ``nvs05`` sets neither, so here
they are callback-free.
"""

from __future__ import annotations

import time
from pathlib import Path

import discopt._rust as _rust
import pytest
from discopt import _timing
from discopt.modeling.core import from_nl

CORPUS = Path(__file__).parent / "data" / "minlplib_nl"

#: Native entry points that are callback-free when no hook is passed (none is, here).
CALLBACK_FREE = (
    "solve_lp_warm_csc_py",
    "solve_milp_csc_py",
    "solve_lp_batch_py",
    "solve_convex_tree_py",
    "solve_milp_relu_csc_py",
)


def test_native_lp_time_is_charged_to_rust(monkeypatch):
    spent = {"calls": 0, "seconds": 0.0}

    def _spy(fn):
        def _wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                spent["calls"] += 1
                spent["seconds"] += time.perf_counter() - t0

        return _wrapped

    for name in CALLBACK_FREE:
        monkeypatch.setattr(_rust, name, _spy(getattr(_rust, name)))

    # nvs05 runs root OBBT, whose probe LPs were the largest uncharged native
    # block on the phase1 panel. The limit only bounds the test's cost: the
    # asserted relation holds at every point of the solve.
    res = from_nl(str(CORPUS / "nvs05.nl")).solve(time_limit=4.0)

    assert spent["calls"] > 0, "no native LP was observed; the spy measured nothing"
    rust_self = res.rust_time - (res.pounce_time or 0.0)
    # The spy's own cost per call (two perf_counter() calls, a closure, and the
    # charge region's enter/exit) is microseconds; 100 us/call is generous. Post-fix
    # the charged time covers the measured time with no slack at all (1-3 ms of
    # margin per solve). An earlier 5 ms/call let a 1.5 s pre-fix leak pass on a
    # machine whose native calls average 3-5 ms (review of #2292).
    slack = 1e-4 * spent["calls"] + 0.05
    assert spent["seconds"] <= rust_self + slack, (
        f"{spent['seconds']:.3f} s inside {spent['calls']} native LP/MILP calls but only "
        f"{rust_self:.3f} s charged to rust: native time is leaking into python_time"
    )


def test_interpreted_region_is_excluded_from_its_native_parent():
    before = _timing.snapshot()
    with _timing.charge("rust"):
        _spin(0.05)
        with _timing.interpreted():
            _spin(0.1)
            with _timing.charge("pounce"):
                _spin(0.05)
    spent = _timing.since(before)
    assert 0.045 <= spent["rust"] < 0.09, f"interpreted callback billed to rust: {spent}"
    assert spent["pounce"] >= 0.045, f"nested charge inside interpreted() lost: {spent}"


def test_as_interpreted_wraps_and_passes_none_through():
    assert _timing.as_interpreted(None) is None
    before = _timing.snapshot()
    hook = _timing.as_interpreted(lambda x: (_spin(0.1), x + 1)[1])
    with _timing.charge("rust"):
        assert hook(1) == 2
    assert _timing.since(before)["rust"] < 0.05


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
