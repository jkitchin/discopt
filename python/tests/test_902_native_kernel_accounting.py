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
def test_native_kernel_seed_time_is_charged_to_jax(monkeypatch):
    """The seed phase (NLP relaxation + sub-NLPs + verification) is JAX work.

    On nvs19 it was the single largest cost in the solve — 12 s of a 20 s wall — and
    it was reported as zero.
    """
    calls = {"seed": 0, "native": 0}
    result = _instrumented_native_solve(monkeypatch, calls=calls)

    assert calls["seed"] == 1, f"the seed phase never ran (calls={calls})"
    assert result.jax_time >= _SEED_SLEEP_S, (
        f"jax_time={result.jax_time!r} excludes the seed phase (slept {_SEED_SLEEP_S}s inside it)"
    )


@pytest.mark.smoke
def test_native_kernel_time_split_is_consistent(monkeypatch):
    """``python_time`` stays the residual of the split, and no bucket goes negative.

    Adding time to two of the three buckets is only an improvement if the third is
    re-derived from them; leaving ``python_time`` computed from the stale totals
    would double-count the kernel's wall.
    """
    calls = {"seed": 0, "native": 0}
    result = _instrumented_native_solve(monkeypatch, calls=calls)

    assert calls["native"] == 1 and calls["seed"] == 1
    assert result.rust_time >= 0.0
    assert result.jax_time >= 0.0
    assert result.python_time >= 0.0, (
        f"python_time={result.python_time!r} went negative — the rust/jax buckets "
        f"double-count (wall={result.wall_time!r})"
    )
    assert result.python_time == pytest.approx(
        result.wall_time - result.rust_time - result.jax_time, abs=1e-9
    )
    # And the accounted work cannot exceed the wall it was measured inside.
    assert result.rust_time + result.jax_time <= result.wall_time + 1e-9
