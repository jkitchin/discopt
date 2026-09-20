"""Issue #1370 Part B: compressed derivatives over declared blocks.

What these pin down is not the speedup — that is measured in
``docs/dev/1370-block-eval-entry-2026-09-20.md`` — but the two things that
would make a speedup worthless: values that differ from the default
evaluator's, and a path that silently does nothing (or silently does
something) when the flag is off.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._block_eval import (
    BLOCK_VECTOR_EVAL_ENV,
    CompressedBlockEvaluator,
    CompressionRefused,
    block_vector_eval_requested,
    build_compressed_evaluator,
    maybe_wrap_evaluator,
)
from discopt._tape_nlp_evaluator import TapeNLPEvaluator
from discopt.modeling.core import Model

pytest.importorskip("jax")

H_STEP = 0.1


def block_model(K: int = 4, steps: int = 6, dim: int = 3, *, declare: bool = True) -> Model:
    """K identical dynamic blocks over a shared parameter vector.

    Coupled across time (``z[t] * z[t+1]``) so the Lagrangian Hessian has
    off-diagonal entries: a diagonal Hessian would let a broken coloring pass.
    """
    m = Model(f"blocks_{K}")
    w = m.continuous("w", shape=(dim,), lb=-2.0, ub=2.0)
    z = m.continuous("z", shape=(K, steps, dim), lb=-10.0, ub=10.0)
    zc = z[:, :-1, :]
    rhs = -w[None, None, :] * zc + 0.1 * dm.sin(zc) + 0.05 * zc * z[:, 1:, :]
    m.subject_to(z[:, 1:, :] - zc - H_STEP * rhs == 0.0, name="dyn")
    m.minimize(dm.sum((z - 1.0) ** 2) + 0.01 * dm.sum(w**2))
    if declare:
        labels = np.repeat(np.arange(K, dtype=np.int64), steps * dim).reshape(K, steps, dim)
        m.set_block(z, labels)
        m.set_block(w, -1)
    return m


def _point(ev, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=ev.n_variables), rng.normal(size=ev.n_constraints)


class TestValuesAgree:
    def test_jacobian_and_hessian_match_the_default_evaluator(self):
        model = block_model()
        base = TapeNLPEvaluator(model)
        comp = build_compressed_evaluator(model, base)

        # The Hessian must actually have off-diagonal structure, or this test
        # would pass on a coloring that only ever recovers a diagonal.
        hr, hc = base.hessian_structure()
        assert int(np.sum(np.asarray(hr) != np.asarray(hc))) > 0

        checked = 0
        for seed in (0, 1, 2):
            x, lam = _point(base, seed)
            bj = base.evaluate_jacobian_values(x)
            cj = comp.evaluate_jacobian_values(x)
            bh = base.evaluate_hessian_values(x, 1.0, lam)
            ch = comp.evaluate_hessian_values(x, 1.0, lam)
            assert bj.shape == cj.shape and bh.shape == ch.shape
            assert np.max(np.abs(bj - cj)) < 1e-10
            assert np.max(np.abs(bh - ch)) < 1e-8
            checked += bj.size + bh.size
        assert checked > 0, "compared nothing — this test would pass on a no-op evaluator"

    def test_compression_is_real(self):
        """Colors must be far fewer than columns, or nothing is being compressed."""
        model = block_model(K=16, steps=6, dim=3)
        base = TapeNLPEvaluator(model)
        comp = build_compressed_evaluator(model, base)
        assert comp.n_jacobian_colors < base.n_variables / 10
        assert comp.n_hessian_seeds < base.n_variables / 10

    def test_seed_counts_grow_far_slower_than_the_blocks(self):
        """The point of the whole exercise: cost that does not track K.

        Not *constant*: ``sparse_hessian.build_hessian_coloring`` gives every
        column whose nnz exceeds a ``sqrt(n)`` threshold its own seed, and which
        of the shared columns cross that threshold shifts a little as n grows —
        measured 7 seeds at K=2 against 11 at K=16 here, and 21 at both K=8 and
        K=32 on the larger models in the entry-experiment doc. What matters is
        that an 8x growth in columns does not buy an 8x growth in passes.
        """
        small = block_model(K=2)
        large = block_model(K=16)
        c_small = build_compressed_evaluator(small, TapeNLPEvaluator(small))
        c_large = build_compressed_evaluator(large, TapeNLPEvaluator(large))
        assert TapeNLPEvaluator(large).n_variables >= 7 * TapeNLPEvaluator(small).n_variables
        assert c_large.n_jacobian_colors == c_small.n_jacobian_colors
        assert c_large.n_hessian_seeds < 2 * c_small.n_hessian_seeds

    def test_everything_else_is_delegated(self):
        model = block_model()
        base = TapeNLPEvaluator(model)
        comp = build_compressed_evaluator(model, base)
        x, _ = _point(base)
        assert comp.n_variables == base.n_variables
        assert comp.n_constraints == base.n_constraints
        assert comp.evaluate_objective(x) == base.evaluate_objective(x)
        assert np.array_equal(comp.evaluate_constraints(x), base.evaluate_constraints(x))
        assert np.array_equal(comp.evaluate_gradient(x), base.evaluate_gradient(x))
        # Structures especially: the solver indexes the values with these, so a
        # second source for them is how a wrong matrix gets built out of right
        # numbers.
        for mine, theirs in zip(comp.jacobian_structure(), base.jacobian_structure()):
            assert np.array_equal(mine, theirs)
        for mine, theirs in zip(comp.hessian_structure(), base.hessian_structure()):
            assert np.array_equal(mine, theirs)


class TestAdmission:
    def test_a_disagreeing_path_is_refused_not_served(self, monkeypatch):
        """The admission check is the whole safety story; prove it can fail."""
        import discopt._block_eval as be

        real = be._make_jac_values_fn

        def sabotaged(cons_fn, colors, seed, rows, cols):
            fn = real(cons_fn, colors, seed, rows, cols)

            def wrong(x, params):
                out = fn(x, params)
                out = np.array(out, copy=True)
                if out.size:
                    out[0] += 1.0  # one entry, one unit: the smallest real error
                return out

            return wrong

        monkeypatch.setattr(be, "_make_jac_values_fn", sabotaged)
        model = block_model()
        base = TapeNLPEvaluator(model)
        with pytest.raises(CompressionRefused, match="disagrees"):
            build_compressed_evaluator(model, base)

    def test_a_refusal_falls_back_to_the_base_evaluator(self, monkeypatch):
        import discopt._block_eval as be

        monkeypatch.setenv(BLOCK_VECTOR_EVAL_ENV, "1")
        monkeypatch.setattr(
            be,
            "build_compressed_evaluator",
            lambda *a, **k: (_ for _ in ()).throw(CompressionRefused("no")),
        )
        model = block_model()
        base = TapeNLPEvaluator(model)
        assert maybe_wrap_evaluator(model, base) is base

    def test_model_without_constraints_is_refused(self):
        m = Model("unconstrained")
        x = m.continuous("x", shape=(4,), lb=-5, ub=5)
        m.minimize(dm.sum(x**2))
        m.set_block(x, [0, 0, 1, 1])
        base = TapeNLPEvaluator(m)
        with pytest.raises(CompressionRefused):
            build_compressed_evaluator(m, base)


class TestGate:
    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv(BLOCK_VECTOR_EVAL_ENV, raising=False)
        assert block_vector_eval_requested() is False
        model = block_model()
        base = TapeNLPEvaluator(model)
        assert maybe_wrap_evaluator(model, base) is base

    def test_zero_is_off(self, monkeypatch):
        monkeypatch.setenv(BLOCK_VECTOR_EVAL_ENV, "0")
        assert block_vector_eval_requested() is False

    def test_on_wraps_a_declared_model(self, monkeypatch):
        monkeypatch.setenv(BLOCK_VECTOR_EVAL_ENV, "1")
        model = block_model()
        base = TapeNLPEvaluator(model)
        wrapped = maybe_wrap_evaluator(model, base)
        assert isinstance(wrapped, CompressedBlockEvaluator)

    def test_on_but_undeclared_model_is_untouched(self, monkeypatch):
        """No declaration, no wrap — the feature is opt-in twice over."""
        monkeypatch.setenv(BLOCK_VECTOR_EVAL_ENV, "1")
        model = block_model(declare=False)
        base = TapeNLPEvaluator(model)
        assert maybe_wrap_evaluator(model, base) is base


class TestSolveIsUnchanged:
    def test_the_nlp_solves_to_the_same_answer(self):
        """A derivative-engine change may not move what the solve returns.

        Through ``solve_nlp`` rather than ``Model.solve``: these blocks are
        nonconvex, so the full entry point runs spatial branch-and-bound over
        the whole tree, which is not what this change touches and takes minutes.
        """
        from discopt.solvers import SolveStatus
        from discopt.solvers.nlp_pounce import solve_nlp

        model = block_model(K=3, steps=5, dim=3)
        base = TapeNLPEvaluator(model)
        comp = build_compressed_evaluator(model, base)
        x0 = np.full(base.n_variables, 0.1)

        a = solve_nlp(base, x0, options={"print_level": 0})
        b = solve_nlp(comp, x0, options={"print_level": 0})

        assert a.status == b.status == SolveStatus.OPTIMAL
        assert abs(a.objective - b.objective) < 1e-9
        assert np.max(np.abs(a.x - b.x)) < 1e-7
        assert a.iterations == b.iterations

    def test_the_solve_path_consults_the_gate(self, monkeypatch):
        """The wiring in ``_solve_continuous`` is reached, with the real evaluator.

        Calls ``_solve_continuous`` directly for the same reason as above.
        """
        import time as _time

        import discopt._block_eval as be
        from discopt.solver import _solve_continuous

        seen: dict = {}
        real = be.maybe_wrap_evaluator

        def recording(model, base):
            seen["model"] = model
            seen["n"] = base.n_variables
            return real(model, base)

        monkeypatch.setattr(be, "maybe_wrap_evaluator", recording)
        monkeypatch.setenv(BLOCK_VECTOR_EVAL_ENV, "1")

        model = block_model(K=2, steps=4, dim=2)
        result = _solve_continuous(model, 60.0, None, _time.perf_counter(), nlp_solver="pounce")

        assert seen["model"] is model
        assert seen["n"] == TapeNLPEvaluator(model).n_variables
        assert result.status == "optimal"


class TestTimingAttribution:
    def test_the_compressed_work_is_charged_to_jax_and_the_rest_to_rust(self):
        """An evaluator that lies about its layer breaks the profile that judges it.

        ``_IpoptCallbacks`` charges every callback to the evaluator's
        ``timing_bucket``. Most callbacks here still run the base evaluator's
        Rust tape, so the bucket must stay the base's; only the two compressed
        callbacks open a ``jax`` frame inside it.
        """
        from discopt import _timing
        from discopt.solvers.nlp_pounce import solve_nlp

        model = block_model(K=3, steps=5, dim=3)
        base = TapeNLPEvaluator(model)
        comp = build_compressed_evaluator(model, base)
        assert comp.timing_bucket == base.timing_bucket == "rust"

        x0 = np.full(base.n_variables, 0.1)
        before = _timing.snapshot()
        solve_nlp(comp, x0, options={"print_level": 0})
        buckets = _timing.since(before)

        # Both must be non-zero: all-rust would mean the compressed path never
        # ran, all-jax would mean the delegated tape calls were misattributed.
        assert buckets.get("jax", 0.0) > 0.0
        assert buckets.get("rust", 0.0) > 0.0


class TestDefaultPathIsUntouched:
    def test_a_default_solve_still_imports_no_jax(self):
        """The gate's import sits on the solve path; it must not drag JAX in.

        ``solve_nlp_from_model`` now imports ``discopt._block_eval`` on every
        call. That module imports numpy and the standard library only, and
        reaches its JAX-using builder solely behind the flag — so the standing
        "an ordinary nonlinear solve imports zero jax modules" property holds.
        Measured in a subprocess, because this test session has already imported
        JAX for everything above.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            import discopt.modeling as dm
            from discopt import Model
            from discopt.solvers.nlp_pounce import solve_nlp_from_model

            m = Model("t")
            x = m.continuous("x", shape=(3,), lb=-5, ub=5)
            m.minimize(dm.sum(dm.exp(x)))
            m.subject_to(dm.sum(x) >= 1)
            before = sum(1 for k in sys.modules if k.split(".")[0] == "jax")
            r = solve_nlp_from_model(m)
            after = sum(1 for k in sys.modules if k.split(".")[0] == "jax")
            print(f"{r.status.name} {before} {after}")
            """
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        status, before, after = out.stdout.strip().split()[-3:]
        assert status == "OPTIMAL"
        assert int(before) == 0 and int(after) == 0, out.stdout
