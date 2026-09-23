"""#1446 -- a Scholtes homotopy must refuse relations it did not lower itself.

``reformulate_scholtes`` skips a relation the model already carries rows for
(``_pending``). For an idempotent re-lower that is right: the rows are there and
are the same rows. For the *homotopy* it is fatal and silent -- the ``f*g <= t``
row is never emitted, so nothing in the model depends on ``t``. The continuation
then drives a parameter no constraint reads: every stage returns the same point
and each is recorded ``accepted=True, reason="subsolver converged",
certified=True``, because each subsolve genuinely did converge, on the
*unregularized* problem.

``Model.complementarity`` lowers on the spot (``gdp``/``sos1``), so the mistake
is a one-liner to make. It already refuses ``method="scholtes"`` itself; this
closes the reverse direction.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt.mpec import complementarity as mpec_complementarity
from discopt.mpec import solve_mpec

#: min (x0-2)^2 + (x1-1)^2  s.t.  0 <= x0 _|_ x1 >= 0,  x0+x1 <= 3,  x in [0,3]^2.
#: Complementarity forces x0 == 0 or x1 == 0:
#:   x1 = 0 branch -> x0 = 2, objective 1.0      <- the optimum
#:   x0 = 0 branch -> x1 = 1, objective 4.0
TRUE_OPTIMUM = 1.0


def _build(*, eager: bool):
    m = Model("mpec1446")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=3.0)
    m.subject_to(x[0] + x[1] <= 3.0)
    pair = (
        m.complementarity(x[0], x[1], name="comp")
        if eager
        else mpec_complementarity(x[0], x[1], name="comp")
    )
    m.minimize((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2)
    return m, x, pair


class TestAPreLoweredRelationIsRefused:
    def test_scholtes_refuses_an_eagerly_lowered_pair(self):
        m, _x, pair = _build(eager=True)
        with pytest.raises(ValueError) as exc:
            solve_mpec(m, [pair], method="scholtes")
        msg = str(exc.value)
        assert "scholtes" in msg
        assert "comp" in msg, msg
        # Names the method that actually lowered it, so the caller can find it.
        assert pair.lowering_in(m) in msg, msg
        # Points at the call that fixes it.
        assert "discopt.mpec.complementarity" in msg, msg

    def test_the_refusal_names_every_clashing_relation(self):
        m = Model("two")
        x = m.continuous("x", shape=(4,), lb=0.0, ub=3.0)
        p1 = m.complementarity(x[0], x[1], name="alpha")
        p2 = m.complementarity(x[2], x[3], name="beta")
        m.minimize(x[0] + x[1] + x[2] + x[3])
        with pytest.raises(ValueError) as exc:
            solve_mpec(m, [p1, p2], method="scholtes")
        msg = str(exc.value)
        assert "alpha" in msg and "beta" in msg, msg
        assert "2 relation" in msg, msg

    def test_the_model_is_not_mutated_by_the_refusal(self):
        """Refuse before touching the model: no half-added parameter or rows."""
        m, _x, pair = _build(eager=True)
        rows_before = len(m._constraints)
        params_before = [p.name for p in m._parameters]
        with pytest.raises(ValueError):
            solve_mpec(m, [pair], method="scholtes")
        assert len(m._constraints) == rows_before
        assert [p.name for p in m._parameters] == params_before
        assert "_mpec_t" not in params_before


class TestTheSupportedPathReachesTheTrueOptimum:
    def test_a_factory_built_pair_is_regularized_and_solved(self):
        m, _x, pair = _build(eager=False)
        assert pair.lowering_in(m) is None, "the factory must not lower"

        r = solve_mpec(m, [pair], method="scholtes")

        # The homotopy's own row must exist -- this is what was missing.
        assert any(getattr(c, "name", None) == "comp_reg" for c in m._constraints), [
            getattr(c, "name", None) for c in m._constraints
        ]
        assert r.status == "local_optimal", r.status
        assert r.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-5)

        # And the returned point actually satisfies complementarity.
        xs = np.asarray(r.x["x"], float).ravel()
        assert abs(float(xs[0] * xs[1])) < 1e-5, xs
        assert xs[0] >= -1e-9 and xs[1] >= -1e-9

    def test_the_continuation_residual_matches_the_admitted_scale(self):
        """The trace must not claim convergence the point does not have."""
        m, _x, pair = _build(eager=False)
        r = solve_mpec(m, [pair], method="scholtes")
        rep = r.mpec_report
        assert rep is not None
        scale = rep.continuation.admitted_residual_scale
        assert rep.complementarity.value <= max(scale, 1e-4) + 1e-6, (
            f"source complementarity {rep.complementarity.value} exceeds the admitted scale {scale}"
        )


class TestTheGuardDoesNotOverreach:
    @pytest.mark.parametrize("method", ["sos1", "gdp"])
    def test_the_exact_lowerings_are_unaffected(self, method):
        """They lower the pair themselves and never consult t; nothing changes."""
        m, _x, pair = _build(eager=True)
        r = solve_mpec(m, [pair], method=method, time_limit=120.0)
        assert r.status in ("optimal", "feasible"), r.status
        assert r.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-5)

    def test_a_second_scholtes_call_still_fails_loudly(self):
        """Already loud before #1446 (duplicate '_mpec_t'); keep it that way."""
        m, _x, pair = _build(eager=False)
        solve_mpec(m, [pair], method="scholtes")
        assert pair.lowering_in(m) == "scholtes"
        with pytest.raises(ValueError):
            solve_mpec(m, [pair], method="scholtes")

    def test_an_unlowered_pair_on_a_fresh_model_is_not_flagged(self):
        m, _x, pair = _build(eager=False)
        assert pair.lowering_in(m) is None
        solve_mpec(m, [pair], method="scholtes")  # no raise
