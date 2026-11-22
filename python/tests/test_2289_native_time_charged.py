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
    # Upper bound 0.12, not 0.09: a runner descheduled for ~40 ms during the 50 ms
    # spin must not fail this. Billing the 0.1 s callback to rust would give >= 0.15.
    assert 0.045 <= spent["rust"] < 0.12, f"interpreted callback billed to rust: {spent}"
    assert spent["pounce"] >= 0.045, f"nested charge inside interpreted() lost: {spent}"


def test_as_interpreted_wraps_and_passes_none_through():
    assert _timing.as_interpreted(None) is None
    before = _timing.snapshot()
    hook = _timing.as_interpreted(lambda x: (_spin(0.1), x + 1)[1])
    with _timing.charge("rust"):
        assert hook(1) == 2
    assert _timing.since(before)["rust"] < 0.05


def test_lazy_callback_time_is_not_billed_to_rust():
    """A native solve that calls back into Python keeps the callback in python_time.

    ``solve_milp_lazy_csc_py`` runs inside ``charge("rust")`` and calls the lazy
    separator for every integer-feasible point. The separator here spins 0.2 s per
    call (the solve makes one or two). If it were not wrapped in
    ``as_interpreted``, that time would be Rust self time and the rust bucket would
    hold at least ``calls * 0.2`` s.
    """
    import numpy as np
    from discopt.solvers.milp_simplex import solve_milp_with_lazy_cuts

    calls = {"n": 0}

    def _lazy(x):
        calls["n"] += 1
        _spin(0.2)
        return []  # accept every point

    before = _timing.snapshot()
    t0 = time.perf_counter()
    res = solve_milp_with_lazy_cuts(
        c=np.array([-1.0, -1.0]),
        A_ub=np.array([[2.0, 1.0], [1.0, 2.0]]),
        b_ub=np.array([4.0, 4.0]),
        bounds=[(0.0, 3.0), (0.0, 3.0)],
        integrality=np.array([1, 1]),
        lazy_callback=_lazy,
        time_limit=30.0,
    )
    wall = time.perf_counter() - t0
    spent = _timing.since(before)

    assert res.objective is not None
    assert calls["n"] > 0, "the lazy callback never ran; the test measured nothing"
    in_callbacks = 0.2 * calls["n"]
    assert spent["rust"] > 0.0, "the lazy MILP solve was not charged to rust at all"
    assert spent["rust"] <= wall - in_callbacks + 0.01, (
        f"rust={spent['rust']:.3f} s of a {wall:.3f} s solve whose {calls['n']} lazy "
        f"callbacks spun {in_callbacks:.3f} s: callback time was billed to rust"
    )


#: Every native solve entry point the extension exports, whether or not anything
#: calls it today: a call site added later is checked without editing this test.
NATIVE_SOLVES = tuple(n for n in dir(_rust) if n.startswith("solve_") and n.endswith("_py"))


def _gap_milp():
    """A small generalized-assignment MILP with marked coupling rows, so
    ``lagrangian_bound=True`` attaches the native tree's ``node_bound_hook``."""
    import discopt.modeling as dm
    import numpy as np

    K, N = 2, 6
    rng = np.random.default_rng(0)
    cost = rng.integers(1, 10, (K, N))
    w = rng.integers(3, 9, (K, N))
    m = dm.Model("gap2289")
    x = m.binary("x", shape=(K * N,))
    m.minimize(sum(int(cost[k, i]) * x[k * N + i] for k in range(K) for i in range(N)))
    for k in range(K):
        cap = int(w[k].sum() * 0.6)
        m.subject_to(sum(int(w[k, i]) * x[k * N + i] for i in range(N)) <= cap)
    for i in range(N):
        c = sum(x[k * N + i] for k in range(K)) == 1
        m.subject_to(c)
        m.mark_coupling(c)
    return m


def test_every_native_solve_is_charged_and_every_callback_is_wrapped(monkeypatch):
    """Structural form of the guard, with no timing in it, so it cannot flake.

    At every call into a native ``solve_*_py`` entry point, (1) the innermost open
    ``_timing`` region must be a ``charge("rust")`` and (2) every callable argument
    must be an ``as_interpreted`` wrapper. (1) is the #2289 leak: a missing region
    sends native time to ``python_time``. (2) is its mirror: a bare callback bills
    interpreted Python to rust. The timing tests above can only cover the
    callback-free entry points; this one covers the callback-taking ones too.

    The solves are picked so that every callback-taking call site runs at least
    once, and the test asserts that each one did: ``dispatch`` reaches the native
    spatial kernel with its ``primal_hook`` and ``incumbent_value``; ``gbd`` reaches
    the lazy MILP with its separator, ``debug_hook`` and ``node_callback``; the GAP
    MILP on the Rust engine reaches ``solve_milp_csc_py`` with a user
    ``node_callback`` (passed as ``debug_hook``) and the Lagrangian
    ``node_bound_hook``; ``nvs05`` covers root OBBT's probe LPs.
    """
    seen: dict[str, set] = {}
    violations: list[str] = []

    def _spy(name, fn):
        def _wrapped(*args, **kwargs):
            stack = _timing._stack()
            if not stack or stack[-1].bucket != "rust":
                top = stack[-1].bucket if stack else "no region"
                violations.append(f"{name} called outside charge('rust') (innermost: {top})")
            for key, val in list(enumerate(args)) + list(kwargs.items()):
                if callable(val):
                    seen.setdefault(name, set()).add(key)
                    if not getattr(val, _timing.INTERPRETED_MARKER, False):
                        violations.append(f"{name}: callback {key!r} is not as_interpreted")
            seen.setdefault(name, set())
            return fn(*args, **kwargs)

        return _wrapped

    for name in NATIVE_SOLVES:
        monkeypatch.setattr(_rust, name, _spy(name, getattr(_rust, name)))

    for inst in ("dispatch", "gbd", "nvs05"):
        from_nl(str(CORPUS / f"{inst}.nl")).solve(time_limit=3.0)
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")  # a pure MILP goes to HiGHS otherwise
    _gap_milp().solve(time_limit=30, lagrangian_bound=True, node_callback=lambda ctx, m: None)

    assert not violations, "\n".join(sorted(set(violations)))
    expected = {
        "solve_lp_warm_csc_py": set(),
        "solve_spatial_tree_py": {"primal_hook", "incumbent_value"},
        "solve_milp_lazy_csc_py": {11, "debug_hook", "node_callback"},
        "solve_milp_csc_py": {"debug_hook", "node_bound_hook"},
    }
    for name, callbacks in expected.items():
        assert name in seen, f"{name} was never called; the probe did not reach it"
        assert callbacks <= seen[name], (
            f"{name} saw callbacks {sorted(map(str, seen[name]))}, expected at least "
            f"{sorted(map(str, callbacks))}: a call site this test guards was not reached"
        )


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
