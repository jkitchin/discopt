"""#863: every root-presolve pass must honour ``time_limit_ms``, not just the loop.

The Rust orchestrator checks its time budget **between passes**
(``presolve/orchestrator.rs``), so a pass whose own internal fixed-point loop does
not poll ``ctx.deadline`` runs to completion regardless of the budget. ``solve_model``
relies on the opposite (its comment reads "the Rust side honours ``time_limit_ms``
between sweeps, so the overrun is bounded by a single sweep"), which is only true
while every long pass polls.

Measured on ``watercontamination0202`` (106,711 vars / 107,209 constraints) against a
7.5 s budget, one pass at a time — **five of ten passes overran**:

    pass                        before              after
    eliminate                   >90 s  (>12x)       7.53 s  1.0x  TimeBudget
    aggregate                   >90 s  (>12x)       7.56 s  1.0x  TimeBudget
    simplify                    >90 s  (>12x)       7.52 s  1.0x  TimeBudget
    fbbt                        >90 s  (>12x)       7.57 s  1.0x  TimeBudget
    probing                     >90 s  (>12x)       7.59 s  1.0x  TimeBudget
    factorable_elim             7.51 s  1.0x        7.52 s  1.0x  (already polled)
    redundancy                  7.12 s  0.9x        7.01 s  0.9x
    implied_bounds              0.09 s              0.09 s
    cliques                     0.04 s              0.04 s
    coefficient_strengthening   0.02 s              0.02 s

(">90 s" is where the probe was killed; they were still running.) ``probing`` is the
instructive one: it *already* polled ``ctx.deadline`` — once per binary variable — but
this instance has 107,209 constraints and only **7** binaries, and each binary costs
two full FBBT runs. The poll was there and the granularity made it useless.

Two levels of test here, and they prove different things:

* the in-repo instances are small enough that these passes finish quickly whether or
  not they poll, so those tests are a *guard* (they catch a pass that ignores the
  budget outright, and a future pass that grows a fixed-point loop), not a
  demonstration of the fix;
* ``test_eliminate_honours_its_budget_on_a_large_instance`` is the decisive one and
  it needs the full MINLPLib snapshot, so it skips when that is absent. The
  authoritative fails-before evidence is the Rust unit tests
  (``presolve::{eliminate,aggregate,fbbt,simplify}::tests::*deadline*``) plus the
  measurement above.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"

# Every pass the root pipeline can enable. A pass that ignores the deadline shows up
# as a wall far above the budget on a big model; here it shows up as work done under
# an already-expired budget.
_PASSES = [
    "eliminate",
    "factorable_elim",
    "aggregate",
    "redundancy",
    "simplify",
    "coefficient_strengthening",
    "implied_bounds",
    "fbbt",
    "probing",
    "cliques",
]


def _repr_for(name: str):
    from discopt._rust import model_to_repr

    model = dm.from_nl(str(_NL_DIR / f"{name}.nl"))
    return model_to_repr(model, getattr(model, "_builder", None))


def _presolve(repr_, passes, time_limit_ms):
    return repr_.presolve(
        passes=passes,
        max_iterations=16,
        time_limit_ms=time_limit_ms,
        work_unit_budget=0,
        fbbt_max_iter=10,
        fbbt_tol=1e-9,
        reduced_cost_info=None,
    )


@pytest.mark.parametrize("pass_name", _PASSES)
def test_each_pass_returns_promptly_under_a_one_millisecond_budget(pass_name):
    """A 1 ms budget must not buy more than a moment of work from any single pass.

    ``eliminate`` failed this before the fix: on a large model its fixed-point loop
    never looked at the deadline. 5 s is a deliberately loose ceiling — the point is
    to catch a pass that ignores the budget entirely, not to measure it.
    """
    repr_ = _repr_for("4stufen")
    t = time.perf_counter()
    _new_repr, raw = _presolve(repr_, [pass_name], 1)
    elapsed = time.perf_counter() - t
    assert elapsed < 5.0, (
        f"pass {pass_name!r} took {elapsed:.2f}s under a 1 ms budget; it is not "
        "polling ctx.deadline inside its own loop"
    )
    assert raw["iterations"] >= 1


def test_eliminate_under_an_expired_budget_leaves_the_model_valid():
    """Bailing early must perform FEWER eliminations, never wrong ones: the model
    that comes back keeps the not-yet-eliminated constraints, so it stays equivalent
    to the input. Compare constraint counts against an unbudgeted run."""
    unbudgeted = _presolve(_repr_for("4stufen"), ["eliminate"], 0)
    budgeted = _presolve(_repr_for("4stufen"), ["eliminate"], 1)
    n_unbudgeted = unbudgeted[0].n_constraints
    n_budgeted = budgeted[0].n_constraints
    assert n_budgeted >= n_unbudgeted, (
        f"the budgeted run removed MORE constraints ({n_budgeted} left) than the "
        f"unbudgeted one ({n_unbudgeted} left); bailing must only do less work"
    )


def test_a_generous_budget_gives_the_same_result_as_no_budget():
    """The deadline poll must be the only difference: with a budget far above the
    work required, the presolve outcome must match the unbudgeted run exactly."""
    unbudgeted = _presolve(_repr_for("4stufen"), ["eliminate"], 0)
    generous = _presolve(_repr_for("4stufen"), ["eliminate"], 600_000)
    assert unbudgeted[0].n_constraints == generous[0].n_constraints
    assert unbudgeted[0].n_vars == generous[0].n_vars
    assert unbudgeted[1]["iterations"] == generous[1]["iterations"]


_BIG = Path(
    os.path.expanduser(
        "~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/watercontamination0202.nl"
    )
)


@pytest.mark.slow
@pytest.mark.skipif(not _BIG.exists(), reason="needs the full MINLPLib snapshot")
def test_eliminate_honours_its_budget_on_a_large_instance():
    """The decisive test, on the instance that exposed the bug.

    Before the fix: >90 s against a 7.5 s budget (>12x), still running when killed.
    After: 7.52 s, ratio 1.0x, terminated_by=TimeBudget.
    """
    from discopt._rust import model_to_repr

    model = dm.from_nl(str(_BIG))
    repr_ = model_to_repr(model, getattr(model, "_builder", None))
    budget_ms = 5_000
    t = time.perf_counter()
    _new_repr, raw = _presolve(repr_, ["eliminate"], budget_ms)
    elapsed = time.perf_counter() - t
    assert elapsed < 3.0 * budget_ms / 1000.0, (
        f"eliminate took {elapsed:.1f}s against a {budget_ms / 1000:.1f}s budget "
        f"({elapsed / (budget_ms / 1000):.1f}x); it is not polling ctx.deadline"
    )
    assert raw["terminated_by"] == "TimeBudget"


def test_fbbt_binding_accepts_and_honours_a_time_limit():
    """``PyModelRepr.fbbt`` had no budget parameter at all, and it is what
    ``tighten_root_bounds_with_fbbt`` calls immediately before tree creation — the
    last unguarded step in root setup. On watercontamination0202 it ran >10 minutes
    against a 30 s solve budget (#863).

    ``time_limit_ms=None`` must stay the default so every other caller is unchanged,
    and an already-tiny budget must return the untightened (still valid) box.
    """
    import numpy as np

    repr_ = _repr_for("4stufen")
    lb_full, ub_full = repr_.fbbt(max_iter=20, tol=1e-8)
    lb_none, ub_none = repr_.fbbt(max_iter=20, tol=1e-8, time_limit_ms=None)
    assert np.array_equal(np.asarray(lb_full), np.asarray(lb_none))
    assert np.array_equal(np.asarray(ub_full), np.asarray(ub_none))

    # A generous budget must be indistinguishable from no budget.
    lb_gen, ub_gen = repr_.fbbt(max_iter=20, tol=1e-8, time_limit_ms=600_000)
    assert np.array_equal(np.asarray(lb_full), np.asarray(lb_gen))
    assert np.array_equal(np.asarray(ub_full), np.asarray(ub_gen))

    # A tiny budget returns a valid enclosure of the fully-tightened box.
    lb_tiny, ub_tiny = repr_.fbbt(max_iter=20, tol=1e-8, time_limit_ms=1)
    assert np.all(np.asarray(lb_tiny) <= np.asarray(lb_full) + 1e-9)
    assert np.all(np.asarray(ub_tiny) >= np.asarray(ub_full) - 1e-9)


def test_fbbt_with_cutoff_binding_accepts_a_time_limit():
    """Same for the cutoff variant, whose default must also stay unlimited."""
    import numpy as np

    repr_ = _repr_for("4stufen")
    a = repr_.fbbt_with_cutoff(20, 1e-8)
    b = repr_.fbbt_with_cutoff(20, 1e-8, None, 600_000)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_root_bound_tightening_passes_its_budget_through(monkeypatch):
    """``tighten_root_bounds_with_fbbt`` must forward ``time_limit_ms`` to the
    binding, and must still do its sound integer rounding when FBBT bails."""
    import numpy as np
    from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

    model = dm.from_nl(str(_NL_DIR / "4stufen.nl"))
    repr_ = _repr_for("4stufen")
    seen: list[object] = []
    real = type(repr_).fbbt

    def _spy(self, *a, **k):
        seen.append(k.get("time_limit_ms", "MISSING"))
        return real(self, *a, **k)

    monkeypatch.setattr(type(repr_), "fbbt", _spy, raising=False)
    n = sum(v.size for v in model._variables)
    lb = np.full(n, -1e3)
    ub = np.full(n, 1e3)
    tighten_root_bounds_with_fbbt(model, lb, ub, [], [], model_repr=repr_, time_limit_ms=1234)
    assert seen == [1234], f"budget was not forwarded to the binding: {seen}"


def test_an_infinite_time_limit_reaches_the_fbbt_call_as_unlimited(monkeypatch):
    """A non-finite ``time_limit`` means "no wall-clock cap" and must arrive at the
    FBBT binding as ``time_limit_ms=None``.

    ``int(1000 * inf)`` raises ``OverflowError``, so the first draft of the root-FBBT
    cap turned every explicitly uncapped solve into a crash. Caught by
    ``test_spatial_native_kernel``'s #788 test, which pins the same contract at a
    different call site; pinned here too because this is where it was broken.
    """
    import discopt.solvers._root_presolve as rp

    seen: list[object] = []
    real = rp.tighten_root_bounds_with_fbbt

    def _spy(*a, **k):
        seen.append(k.get("time_limit_ms", "MISSING"))
        return real(*a, **k)

    monkeypatch.setattr(rp, "tighten_root_bounds_with_fbbt", _spy)

    # A bilinear body with an integer: this routes onto the spatial B&B path, which
    # is the one that calls tighten_root_bounds_with_fbbt before tree creation.
    m = dm.Model("inf_budget")
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=1.0, ub=4.0)
    z = m.integer("z", lb=0, ub=2)
    m.minimize(x * y + z)
    m.subject_to(x + y >= 3.0)
    m.subject_to(x - z <= 2.0)
    m.solve(time_limit=float("inf"))

    assert seen, "tighten_root_bounds_with_fbbt was never reached"
    assert seen[0] is None, (
        f"an infinite time_limit reached the FBBT call as {seen[0]!r}; it must be None "
        "(unlimited), and int(1000 * inf) raises OverflowError"
    )


def test_full_root_presolve_respects_its_budget():
    """The pipeline as ``solve_model`` invokes it, with every default pass on."""
    from discopt._relax.presolve_pipeline import run_root_presolve

    t = time.perf_counter()
    _repr, stats = run_root_presolve(
        _repr_for("4stufen"), eliminate=True, fbbt=True, time_limit_ms=1
    )
    elapsed = time.perf_counter() - t
    assert elapsed < 10.0, f"root presolve took {elapsed:.2f}s under a 1 ms budget"
    assert stats["terminated_by"] in {"TimeBudget", "NoProgress", "IterationCap", "Infeasible"}
