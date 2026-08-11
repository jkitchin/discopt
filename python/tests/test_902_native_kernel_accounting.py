"""#902 finding 3: the native spatial kernel reported broken time accounting.

The kernel path handed ``SolveResult`` the ``rust_time``/``jax_time`` counters
*as they stood when the driver was called* — i.e. whatever presolve had
accumulated — and never added its own work. Measured on nvs19 (#902): a ~7 s Rust
tree reported ``rust_time`` ≈ 1e-4 s and ``jax_time`` = 0.0, so ``python_time``
(computed as ``wall - rust - jax``) absorbed the entire solve and the three fields
together said "this solve was pure Python", which is the opposite of the truth.

That is the CLAUDE.md §6 failure mode — an instrument that reads plausibly while
measuring nothing — so these assertions are on *executed* call counts plus
lower bounds injected by deliberate sleeps, never on ambient wall time (§9).
"""

import time

import numpy as np
import pytest
from discopt import Model

_rust = pytest.importorskip("discopt._rust")
if not hasattr(_rust, "solve_spatial_tree_py"):
    pytest.skip("native spatial kernel binding not built", allow_module_level=True)

# Injected durations. Chosen well above scheduler jitter but small enough to keep
# the test in the smoke suite; every assertion is a LOWER bound, so competing load
# can only ever make them more true.
_SEED_SLEEP_S = 0.20
_NATIVE_SLEEP_S = 0.30


def _bilinear_min():
    """min x*y s.t. x+y>=3, x,y in [0,2] -> 2.0. In the kernel's covered subset."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y >= 3)
    m.minimize(x * y)
    return m


def _instrumented_native_solve(monkeypatch, *, calls):
    """Run one native-kernel solve whose seed and Rust tree take a KNOWN time."""
    import discopt.solver as S

    def fake_seed(*args, **kwargs):
        calls["seed"] += 1
        time.sleep(_SEED_SLEEP_S)
        # A genuinely feasible corner of the model above, in internal minimize units.
        return 2.0, np.array([2.0, 1.0])

    def fake_native(**kwargs):
        calls["native"] += 1
        time.sleep(_NATIVE_SLEEP_S)
        return {
            "status": "optimal",
            "incumbent": 2.0,
            "incumbent_x": np.array([2.0, 1.0]),
            "bound": 2.0,
            "node_count": 5,
            "n_lp_solves": 5,
            "n_uncertified": 0,
        }

    monkeypatch.setattr(S, "_native_kernel_seed", fake_seed)
    monkeypatch.setattr(_rust, "solve_spatial_tree_py", fake_native)
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")
    return _bilinear_min().solve(time_limit=30.0, verify_incumbent=False)


@pytest.mark.smoke
def test_native_kernel_time_is_charged_to_rust_not_python(monkeypatch):
    """The Rust tree's wall time must land in ``rust_time``."""
    calls = {"seed": 0, "native": 0}
    result = _instrumented_native_solve(monkeypatch, calls=calls)

    # §6: prove the probe fired. Without this the assertions below would pass
    # vacuously on any run that declined the kernel.
    assert calls["native"] == 1, f"native binding was never reached (calls={calls})"
    assert result.status == "optimal"

    assert result.rust_time >= _NATIVE_SLEEP_S, (
        f"rust_time={result.rust_time!r} excludes the native tree "
        f"(slept {_NATIVE_SLEEP_S}s inside the binding)"
    )


@pytest.mark.smoke
def test_native_kernel_seed_time_is_attributed(monkeypatch):
    """The seed phase must appear in the profile — in *some* bucket, never dropped.

    Supersedes ``test_native_kernel_seed_time_is_charged_to_jax`` (#902), whose
    premise — "the seed phase is JAX work" — measurement contradicted. The seed
    runs an NLP relaxation plus sub-NLPs; the *optimization* is POUNCE's Rust IPM
    and JAX only supplies derivative callbacks. Charging the whole phase to
    ``jax_time`` was an instance of the very bug #902 set out to fix: a bucket
    inflated by another layer's work. Measured corpus-wide, that phase-level
    attribution reported 0.14-1.09 s of ``jax_time`` on nine instances where
    ``jax`` never entered ``sys.modules`` at all.

    #902's real guarantee is preserved and asserted below: the phase's time is
    accounted for, and the buckets still partition the wall. Here the seed is
    replaced by a bare ``time.sleep`` — interpreted Python, not POUNCE and not
    JAX — so it must land in ``python_time``. With the *real* seed it splits
    across ``pounce_time`` and ``jax_time`` per :mod:`discopt._timing`.
    """
    calls = {"seed": 0, "native": 0}
    result = _instrumented_native_solve(monkeypatch, calls=calls)

    assert calls["seed"] == 1, f"the seed phase never ran (calls={calls})"
    # The stubbed seed is a pure-Python sleep, so it belongs to python_time.
    assert result.python_time >= _SEED_SLEEP_S, (
        f"python_time={result.python_time!r} excludes the seed phase "
        f"(slept {_SEED_SLEEP_S}s of pure Python inside it)"
    )
    # And it must not have been mis-charged to a native bucket.
    assert result.pounce_time < _SEED_SLEEP_S, (
        f"pounce_time={result.pounce_time!r} absorbed a pure-Python sleep"
    )


@pytest.mark.smoke
def test_native_kernel_time_split_is_consistent(monkeypatch):
    """``rust_time`` and ``python_time`` partition the wall; subsets stay inside.

    The old identity asserted here was ``python_time == wall - rust - jax``, which
    treated ``jax_time`` as a *peer* of ``python_time``. It is not: ~96 % of JAX's
    cost on the solve path is interpreted Python (measured on heatexch_gen3 as
    13.34 s Python vs 0.56 s native XLA), so that formula subtracted Python time
    twice and the fields could never reconcile.

    The model asserted now:
      * ``rust_time`` + ``python_time`` == ``wall_time``  (disjoint, exhaustive)
      * ``pounce_time`` <= ``rust_time``                  (POUNCE is Rust)
      * ``jax_time``    <= ``python_time``                (JAX time is Python time)
    """
    calls = {"seed": 0, "native": 0}
    result = _instrumented_native_solve(monkeypatch, calls=calls)

    assert calls["native"] == 1 and calls["seed"] == 1
    for field in ("rust_time", "python_time", "jax_time", "pounce_time"):
        assert getattr(result, field) >= 0.0, f"{field}={getattr(result, field)!r} went negative"

    assert result.rust_time + result.python_time == pytest.approx(result.wall_time, abs=1e-6), (
        f"rust_time={result.rust_time!r} + python_time={result.python_time!r} does not "
        f"partition wall_time={result.wall_time!r}"
    )
    assert result.pounce_time <= result.rust_time + 1e-9, (
        f"pounce_time={result.pounce_time!r} exceeds rust_time={result.rust_time!r}; "
        "POUNCE is Rust and must be a subset"
    )
    assert result.jax_time <= result.python_time + 1e-9, (
        f"jax_time={result.jax_time!r} exceeds python_time={result.python_time!r}; "
        "JAX time is Python time and must be a subset"
    )
