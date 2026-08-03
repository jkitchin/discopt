"""Layer-profile attribution: the buckets must describe the solve they measured.

Issue #74. The pre-existing model had three fields — ``rust_time`` / ``jax_time`` /
``python_time`` — accumulated by *phase* timers around large regions of
``solver.py``, with ``python_time`` derived as ``wall - rust - jax``. Two
independent defects, both measured on the in-repo corpus:

1. **Phases were mixed.** One accumulator spanned ~745 lines and enclosed the Rust
   simplex and the numpy relaxation build, charging all of it to JAX. Nine corpus
   instances reported 0.14-1.09 s of ``jax_time`` on solves where ``jax`` never
   entered ``sys.modules`` (``st_testgr3``: 1.09 s of a 1.13 s solve).

2. **The buckets were not disjoint.** ~96 % of JAX's cost on the solve path is
   interpreted Python, so subtracting ``jax_time`` from the wall removed Python
   time twice — and POUNCE (11.69 s of a 15 s ``tspn08`` solve) had no bucket and
   was silently absorbed.

The model asserted here: ``rust_time`` and ``python_time`` are disjoint and
partition the wall; ``pounce_time`` and ``jax_time`` are subsets of those two.

Every test asserts an *executed* count or an injected lower bound rather than
ambient wall time, so competing load can only make them more true (CLAUDE.md §9).
"""

import subprocess
import sys
import textwrap

import pytest
from discopt import Model, _timing

# --------------------------------------------------------------------------
# The accounting primitive
# --------------------------------------------------------------------------


def test_charge_records_self_time_not_wall_time():
    """A nested region's time is subtracted from its parent.

    This is the defining property. POUNCE calls back into Python for every
    derivative, so ``problem.solve()``'s wall contains the evaluator callbacks.
    Charging that wall to ``pounce`` would rebuild the exact defect this module
    exists to prevent.
    """
    before = _timing.snapshot()
    with _timing.charge("pounce"):
        _spin(0.06)
        with _timing.charge("jax"):
            _spin(0.06)
    spent = _timing.since(before)

    assert spent["jax"] >= 0.05, f"nested region not recorded: {spent}"
    # pounce saw ~0.12 s of wall but only ~0.06 s is its own.
    assert spent["pounce"] < 0.11, (
        f"pounce_time={spent['pounce']:.4f} absorbed the nested jax region: {spent}"
    )


def test_charge_is_reentrant_for_the_same_bucket():
    """POUNCE's LP path is reachable from inside an NLP solve; do not double-count."""
    before = _timing.snapshot()
    with _timing.charge("pounce"):
        with _timing.charge("pounce"):
            _spin(0.06)
    spent = _timing.since(before)
    assert 0.05 <= spent["pounce"] < 0.11, f"re-entrant charge double-counted: {spent}"


def test_charge_records_a_raising_boundary():
    """An exception costing real time must not vanish from the profile."""
    before = _timing.snapshot()
    with pytest.raises(RuntimeError):
        with _timing.charge("rust"):
            _spin(0.06)
            raise RuntimeError("boundary failed")
    assert _timing.since(before)["rust"] >= 0.05, "a raising boundary was not charged"


def test_charge_rejects_an_unknown_bucket():
    """A typo'd bucket must fail loudly, not silently measure nothing (§6)."""
    with pytest.raises(ValueError, match="unknown timing bucket"):
        with _timing.charge("jaxx"):
            pass


def _spin(seconds: float) -> None:
    """Busy-wait. ``time.sleep`` would also work, but spinning keeps the thread
    on-CPU so the measurement reflects work rather than scheduler latency."""
    import time

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# --------------------------------------------------------------------------
# End-to-end invariants
# --------------------------------------------------------------------------


def _bilinear_minlp():
    m = Model()
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y + z)
    return m


def _pure_milp():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    z = m.integer("z", lb=0, ub=5)
    m.subject_to(x + z >= 3.0)
    m.minimize(x + 2 * z)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("build", [_pure_milp, _bilinear_minlp], ids=["milp", "minlp"])
def test_buckets_partition_the_wall(build):
    """rust + python == wall, and the diagnostic subsets stay inside their parents."""
    result = build().solve(time_limit=30.0)

    assert result.rust_time >= 0.0 and result.python_time >= 0.0
    assert result.rust_time + result.python_time == pytest.approx(result.wall_time, abs=1e-6), (
        f"buckets do not partition the wall: rust={result.rust_time!r} "
        f"python={result.python_time!r} wall={result.wall_time!r}"
    )
    assert result.pounce_time <= result.rust_time + 1e-9, "POUNCE is Rust; must be a subset"
    assert result.jax_time <= result.python_time + 1e-9, (
        "JAX time is Python time; must be a subset, not a peer"
    )


# --------------------------------------------------------------------------
# The headline regression: no phantom JAX time
# --------------------------------------------------------------------------

_NO_JAX_PROBE = textwrap.dedent(
    """
    import sys
    from discopt import Model

    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    z = m.integer("z", lb=0, ub=5)
    m.subject_to(x + z >= 3.0)
    m.minimize(x + 2 * z)
    r = m.solve(time_limit=30.0)

    # §6: prove the probe fired rather than reporting a vacuous pass.
    assert r.status == "optimal", r.status
    print("JAX_IMPORTED", "jax" in sys.modules)
    print("JAX_TIME", r.jax_time)
    print("WALL", r.wall_time)
    """
)


@pytest.mark.smoke
def test_pure_milp_reports_no_jax_time():
    """A solve that never imports JAX must report ``jax_time == 0``.

    Run in a subprocess so ``sys.modules`` is clean — an in-process check would be
    polluted by any earlier test that imported JAX, and would pass vacuously.

    Before #74 this failed on nine corpus instances; ``st_test1`` reported 0.19 s
    of ``jax_time`` on a 0.22 s solve that never loaded the library.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _NO_JAX_PROBE], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"

    out = dict(line.split(maxsplit=1) for line in proc.stdout.splitlines() if " " in line)
    assert out["JAX_IMPORTED"] == "False", (
        "this model was supposed to stay off the JAX path; the fixture no longer "
        "exercises the regression"
    )
    assert float(out["JAX_TIME"]) == 0.0, (
        f"jax_time={out['JAX_TIME']} on a solve where jax was never imported "
        f"(wall={out['WALL']}) — the layer profile is attributing time to a "
        f"library that was never loaded"
    )
