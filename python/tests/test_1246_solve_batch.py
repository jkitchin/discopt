"""``dm.solve_batch`` — many small independent global solves (#1246).

The CALPHAD decomposition behind #1249 is one small global solve per phase, times
the number of (T, x) conditions in a trace or data points in a fit: tens to
thousands of independent solves. ``solve_batch`` is the public API for that, and
these tests pin the two properties a caller relies on — the results are the same
ones a sequential loop would produce, and one model's failure does not take the
batch down with it.

Measured on this file's 32-model panel (4-core machine, 8 workers):
sequential 5.59 s, parallel 2.69 s, 2.08x, with 0/32 differences in status,
objective or bound.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = [pytest.mark.smoke]


def _bilinear(i: int) -> "dm.Model":
    """A tiny nonconvex global model; the coefficient makes each one distinct."""
    rng = np.random.default_rng(i)
    m = dm.Model(f"batch{i}")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(float(rng.uniform(0.5, 2.0)) * x * y - x - y)
    m.subject_to(x + y <= 1.5)
    return m


def _same(a, b, *, tol=1e-12) -> bool:
    if a.status != b.status:
        return False
    for field in ("objective", "bound"):
        va, vb = getattr(a, field), getattr(b, field)
        if (va is None) != (vb is None):
            return False
        if va is not None and abs(va - vb) > tol:
            return False
    return True


def test_workers_1_matches_solving_each_model_directly():
    """The sequential path is a plain ``m.solve()`` per model — no serialization,
    no copy, so nothing can drift between the two."""
    direct = [_bilinear(i).solve(time_limit=30) for i in range(4)]
    batched = dm.solve_batch([_bilinear(i) for i in range(4)], workers=1, time_limit=30)
    assert len(batched) == 4
    for i, (d, b) in enumerate(zip(direct, batched)):
        assert _same(d, b), (i, d.status, b.status, d.objective, b.objective)


def test_parallel_matches_sequential_and_keeps_order():
    models_seq = [_bilinear(i) for i in range(8)]
    models_par = [_bilinear(i) for i in range(8)]
    seq = dm.solve_batch(models_seq, workers=1, time_limit=30)
    par = dm.solve_batch(models_par, workers=4, time_limit=30)
    assert len(par) == len(seq) == 8
    for i, (a, b) in enumerate(zip(seq, par)):
        assert _same(a, b), (i, a.status, b.status, a.objective, b.objective, a.bound, b.bound)
    # Distinct models give distinct optima, so an out-of-order collection would
    # show up as a mismatch above; assert the objectives really are distinct so
    # that check cannot pass vacuously.
    assert len({round(r.objective, 9) for r in par}) > 1


def test_a_failing_model_does_not_abort_the_batch():
    broken = dm.Model("no_objective")
    broken.continuous("z", lb=0, ub=1)
    results = dm.solve_batch([_bilinear(0), broken, _bilinear(1)], workers=2, time_limit=30)
    assert [r.status for r in results] == ["optimal", "error", "optimal"]
    assert results[1].error is not None
    assert "objective" in results[1].error.lower(), results[1].error
    assert results[0].error is None and results[2].error is None


def test_a_failing_model_is_captured_sequentially_too():
    broken = dm.Model("no_objective_seq")
    broken.continuous("z", lb=0, ub=1)
    results = dm.solve_batch([broken, _bilinear(2)], workers=1, time_limit=30)
    assert [r.status for r in results] == ["error", "optimal"]
    assert results[0].error is not None


def test_parallel_results_carry_their_model_back():
    """``result.value(var)`` needs the model the worker could not send back."""
    models = [_bilinear(i) for i in range(2)]
    results = dm.solve_batch(models, workers=2, time_limit=30)
    for model, result in zip(models, results):
        assert result._model is model
        val = result.value(model._variables[0])
        assert np.asarray(val).shape in ((), (1,))


def test_callbacks_are_refused_for_worker_processes():
    with pytest.raises(ValueError, match="cannot forward"):
        dm.solve_batch(
            [_bilinear(0), _bilinear(1)],
            workers=2,
            node_callback=lambda ctx, model: None,
        )


def test_streaming_is_refused():
    with pytest.raises(ValueError, match="stream=True"):
        dm.solve_batch([_bilinear(0), _bilinear(1)], workers=2, stream=True)


def test_argument_validation():
    with pytest.raises(ValueError, match="positive int"):
        dm.solve_batch([_bilinear(0)], workers=0)
    with pytest.raises(TypeError, match="expected a Model"):
        dm.solve_batch([_bilinear(0), "not a model"], workers=1)


def test_empty_batch_returns_empty_list():
    assert dm.solve_batch([], workers=4) == []


@pytest.mark.slow
def test_acceptance_32_models_identical_and_faster(capsys):
    """#1246's acceptance: 32 independent small global models give identical
    ``status``/``objective``/``bound`` at ``workers=1`` and ``workers=8``, with the
    wall-clock speedup reported (not asserted — the machine decides that)."""
    n = 32
    seq_models = [_bilinear(i) for i in range(n)]
    t0 = time.time()
    seq = dm.solve_batch(seq_models, workers=1, time_limit=60)
    t_seq = time.time() - t0

    par_models = [_bilinear(i) for i in range(n)]
    t0 = time.time()
    par = dm.solve_batch(par_models, workers=8, time_limit=60)
    t_par = time.time() - t0

    compared = 0
    for i, (a, b) in enumerate(zip(seq, par)):
        assert _same(a, b), (i, a.status, b.status, a.objective, b.objective, a.bound, b.bound)
        compared += 1
    assert compared == n, compared
    assert all(r.status == "optimal" for r in par)

    with capsys.disabled():
        print(
            f"\n[#1246] {n} models: workers=1 {t_seq:.2f}s, workers=8 {t_par:.2f}s, "
            f"speedup {t_seq / max(t_par, 1e-9):.2f}x, {compared}/{n} results identical"
        )
