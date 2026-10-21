"""Issue #1870: #1193's native-exit primal polish was starved on ``time_limit`` exits.

nvs19 stalls at -1098.2 against a -1098.4 optimum. #1193 fixed it with a post-kernel
``integer_box_search`` polish, but measured the fix only on ``node_limit`` exits. On
a clock-limited run (every benchmark run) it did nothing, for three measured reasons:

1. **Stale deadline.** The polish was handed the kernel's *base* deadline. The kernel
   reclaims the #917 extension once it holds an incumbent -- the only state in which
   the polish runs -- so on a clock exit that deadline had passed long before (nvs19
   at ``time_limit=20``: 8.4 s before the polish started; 0 cells enumerated).
2. **No slice.** Even the true deadline leaves nothing: the kernel runs to it.
3. **Cost.** The polish ran the feasibility arbiter on every cell (nvs19: 2.5 s for
   6,560 cells, 12 of which can beat the incumbent) and built its evaluator through
   ``cached_evaluator``, which imports JAX into an otherwise JAX-free solve
   (~0.5 s inside the exit).

The fix: an ``objective_cutoff`` that skips the arbiter on cells that cannot win, the
canonical JAX-free evaluator funnel, the solve's true end as the polish deadline, and
a small slice withheld from the kernel (out of the #917 extension first). Measured on
nvs19 at ``time_limit=10/20/30``: -1098.4 on every run, polish 0.035 s.

nvs19 is not in the in-repo corpus, so the class is pinned here on a 5-integer
2-opt trap and on the vendored nvs13 (kernel-routed, pure integer).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from discopt._relax import primal_heuristics as ph
from discopt._relax.primal_heuristics import integer_box_polish_may_fire, integer_box_search
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit

NVS13 = Path(__file__).parent / "data" / "minlplib_nl" / "nvs13.nl"


def _two_opt_trap(*, with_continuous: bool = False) -> Model:
    """The #1193 trap: the origin is a strict 2-opt local optimum; ``(1,1,1,0,0)``
    pays -0.7 three coordinates away."""
    m = Model("two_opt_trap")
    xs = [m.integer(f"x{i}", lb=0, ub=4) for i in range(5)]
    if with_continuous:
        c = m.continuous("c", lb=0.0, ub=1.0)
        m.subject_to(c <= 1.0)
    m.subject_to(sum(xs) <= 3)
    m.minimize(-(xs[0] * xs[1] * xs[2]) + 0.1 * sum(xs))
    return m


ORIGIN = np.zeros(5)


# ---------------------------------------------------------------------------
# objective_cutoff
# ---------------------------------------------------------------------------


def test_cutoff_returns_the_uncut_point_when_it_improves():
    m = _two_opt_trap()
    ev = make_evaluator(m)
    f0 = float(ev.evaluate_objective(ORIGIN))
    uncut = integer_box_search(m, ORIGIN, radius=1, evaluator=ev)
    cut = integer_box_search(m, ORIGIN, radius=1, evaluator=ev, objective_cutoff=f0)
    assert uncut is not None and cut is not None
    np.testing.assert_array_equal(cut[0], uncut[0])
    assert cut[1] == uncut[1] == pytest.approx(-0.7)


def test_cutoff_returns_none_when_nothing_beats_it():
    m = _two_opt_trap()
    ev = make_evaluator(m)
    # The global optimum's own value: no cell is strictly below it. (Evaluated, not
    # the literal -0.7 -- it evaluates to -0.7000000000000001.)
    f_star = float(ev.evaluate_objective(np.array([1.0, 1.0, 1.0, 0.0, 0.0])))
    assert integer_box_search(m, ORIGIN, radius=1, evaluator=ev, objective_cutoff=f_star) is None


def test_cutoff_runs_the_arbiter_only_on_cells_that_can_win(monkeypatch):
    """The point of the cutoff: the arbiter, not the objective, is what a direct
    cell costs. Count arbiter calls in both arms and compare with an independent
    count of the cells whose objective beats the cutoff."""
    m = _two_opt_trap()
    ev = make_evaluator(m)
    f0 = float(ev.evaluate_objective(ORIGIN))
    calls = {"n": 0}
    real = ph._finalize_candidate

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ph, "_finalize_candidate", counting)
    integer_box_search(m, ORIGIN, radius=1, evaluator=ev)
    uncut_calls = calls["n"]
    calls["n"] = 0
    integer_box_search(m, ORIGIN, radius=1, evaluator=ev, objective_cutoff=f0)
    cut_calls = calls["n"]

    import itertools

    winners = 0
    for d in itertools.product((0, 1), repeat=5):  # radius-1 box clamped at lb=0
        x = np.array(d, dtype=float)
        if x.any() and float(ev.evaluate_objective(x)) < f0:
            winners += 1
    assert uncut_calls == 2**5 - 1  # every cell but the incumbent's own
    assert cut_calls == winners
    assert 0 < cut_calls < uncut_calls


def test_cutoff_also_filters_the_sub_nlp_path():
    """Three integers and one continuous slot: the sub-NLP path (cap 3), where the
    cutoff saves nothing but must give the same answer-or-None semantics."""
    m = Model("sub_nlp_trap")
    xs = [m.integer(f"x{i}", lb=0, ub=4) for i in range(3)]
    c = m.continuous("c", lb=0.0, ub=1.0)
    m.subject_to(sum(xs) <= 3)
    m.minimize(-(xs[0] * xs[1] * xs[2]) + 0.1 * sum(xs) + c)
    x0 = np.zeros(4)
    uncut = integer_box_search(m, x0, radius=1)
    assert uncut is not None and uncut[1] == pytest.approx(-0.7, abs=1e-6), uncut
    ev = make_evaluator(m)
    cut = integer_box_search(
        m, x0, radius=1, evaluator=ev, objective_cutoff=float(ev.evaluate_objective(x0))
    )
    assert cut is not None and cut[1] == pytest.approx(uncut[1], abs=1e-9)
    assert integer_box_search(m, x0, radius=1, objective_cutoff=uncut[1] - 1e-6) is None


def test_may_fire_tracks_the_default_caps(monkeypatch):
    assert integer_box_polish_may_fire(_two_opt_trap())  # 5 ints, direct cap 64
    assert not integer_box_polish_may_fire(_two_opt_trap(with_continuous=True))  # sub-NLP cap 3
    no_int = Model("cont")
    y = no_int.continuous("y", lb=0, ub=1)
    no_int.minimize(y)
    assert not integer_box_polish_may_fire(no_int)
    monkeypatch.setenv("DISCOPT_INTEGER_BOX_POLISH", "0")
    assert not integer_box_polish_may_fire(_two_opt_trap())


# ---------------------------------------------------------------------------
# The polish must not drag JAX into a JAX-free solve
# ---------------------------------------------------------------------------


def test_polish_does_not_import_jax():
    code = (
        "import sys, numpy as np, logging; logging.disable(logging.CRITICAL)\n"
        "from discopt.modeling.core import Model\n"
        "from discopt.solver import _native_exit_primal_polish\n"
        "m = Model('t'); xs = [m.integer(f'x{i}', lb=0, ub=4) for i in range(5)]\n"
        "m.subject_to(sum(xs) <= 3); m.minimize(-(xs[0] * xs[1] * xs[2]) + 0.1 * sum(xs))\n"
        "assert 'jax' not in sys.modules, 'precondition: jax loaded before the polish'\n"
        "out = _native_exit_primal_polish(m, np.zeros(5), 0.0, -10.0, 5, None)\n"
        "assert out is not None and abs(out[1] + 0.7) < 1e-9, out\n"
        "print('JAX_LOADED', 'jax' in sys.modules)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_NLP_EVAL"}
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="/", env=env
    )
    assert p.returncode == 0, p.stderr[-2000:]
    assert "JAX_LOADED False" in p.stdout, p.stdout


# ---------------------------------------------------------------------------
# Budget plumbing on a clock-limited kernel exit
# ---------------------------------------------------------------------------


def test_clock_limited_kernel_exit_leaves_the_polish_a_live_deadline(monkeypatch):
    """Drive the real nvs13 solve through the native kernel, then make the kernel
    behave like a clock-limited exit: hold the call until its own full deadline
    (base time limit + the extension it was granted) and report ``time_limit``.
    Before #1870 the polish was called with the stale base deadline -- already in
    the past -- and could enumerate nothing."""
    import discopt.modeling as dm
    from discopt import _rust
    from discopt import solver as S

    seen: dict = {}
    real_kernel = _rust.solve_spatial_tree_py

    def clock_limited_kernel(**kw):
        t_in = time.perf_counter()
        res = real_kernel(**kw)
        full = float(kw.get("time_limit_s") or 0.0) + float(
            kw.get("incumbent_time_extension_s") or 0.0
        )
        seen["kernel_ext"] = kw.get("incumbent_time_extension_s")
        while time.perf_counter() < t_in + full:
            time.sleep(0.01)
        res["status"] = "time_limit"
        return res

    real_polish = S._native_exit_primal_polish

    def recording_polish(model, x_flat, obj_val, bound_val, n_orig, deadline, source=None):
        seen["slack"] = None if deadline is None else deadline - time.perf_counter()
        return real_polish(model, x_flat, obj_val, bound_val, n_orig, deadline, source=source)

    monkeypatch.setattr(_rust, "solve_spatial_tree_py", clock_limited_kernel)
    monkeypatch.setattr(S, "_native_exit_primal_polish", recording_polish)

    r = dm.from_nl(str(NVS13)).solve(time_limit=12.0)
    assert (r.algorithm_route or "").startswith("native-spatial"), r.algorithm_route
    assert "slack" in seen, "the polish never ran -- the probe measured nothing"
    reserve = (r.solver_stats or {}).get("budget/polish_reserve_s")
    assert reserve is not None and reserve > 0.0, r.solver_stats
    # The polish starts with (most of) the withheld slice still ahead of it.
    assert seen["slack"] is not None and seen["slack"] > 0.5 * reserve, (seen, reserve)


def test_continuous_solve_with_duals_does_not_import_jax():
    """The second leak of the same class, found by solving every in-repo instance and
    checking ``sys.modules``: 65 of 66 stay JAX-free; ``chance`` loaded JAX through
    ``_duals_against_declared_box`` -> ``jacobian_for_recovery``, which imported
    ``_relax.nlp_evaluator`` for one integer threshold."""
    chance = Path(__file__).parent / "data" / "minlplib_nl" / "chance.nl"
    code = (
        "import sys, logging, warnings; logging.disable(logging.CRITICAL)\n"
        "warnings.simplefilter('ignore')\n"
        "import discopt.modeling as dm\n"
        f"r = dm.from_nl({str(chance)!r}).solve(time_limit=30)\n"
        "assert r.status == 'optimal', r.status\n"
        "assert r.constraint_duals is not None, 'the probe must reach dual recovery'\n"
        "print('JAX_LOADED', 'jax' in sys.modules)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "DISCOPT_NLP_EVAL"}
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd="/", env=env
    )
    assert p.returncode == 0, p.stderr[-2000:]
    assert "JAX_LOADED False" in p.stdout, p.stdout
