"""Issue #902 — the native-kernel seed must enumerate GENERAL integers, not just 0/1.

The #764 seed enumerated ``itertools.product((0.0, 1.0), repeat=k)`` over ``free_int``,
a list filtered only by *span > 0.5* (i.e. "not pinned by presolve"). On the binary
models it was graduated against (tanksize: 9 integers, every box ``[0,0]``/``[0,1]``/
``[1,1]``) that is the right enumeration. On an all-general-integer model it pins every
variable to 0 or 1 regardless of its declared box, so on nvs17/19/24 (``[0, 200]``
integers) every candidate was a ``{0,1}`` corner point:

    nvs17 seed -312.6 (71.6% off)   nvs19 -315.0 (71.3% off)   nvs24 -292.6 (71.7% off)

and nvs19's -315.0 was *exactly* the incumbent the default-ON kernel reported after
9,303 nodes. The fix brackets each free integer's continuous-relaxation value instead.

These tests pin both halves of the contract:
  * the general-integer case is actually enumerated (the regression), and
  * the binary case is byte-for-byte the enumeration #764 graduated on (no collateral).
"""

import numpy as np
import pytest
from discopt import Model

_rust = pytest.importorskip("discopt._rust")
if not hasattr(_rust, "solve_spatial_tree_py"):
    pytest.skip("native spatial kernel binding not built", allow_module_level=True)

# Card 4b: every solver symbol this file uses now lives in the
# ``native_kernel`` module; two of them are deliberately not re-exported
# from ``discopt.solver``, so import them from their defining module.
from discopt.solver import native_kernel as S  # noqa: E402

# --------------------------------------------------------------------------------
# _native_seed_bracket — the per-variable candidate set
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("x_rel", [0.0, 0.001, 0.25, 0.5, 0.5000001, 0.75, 0.999, 1.0])
def test_bracket_on_binary_box_is_exactly_zero_one(x_rel):
    """A ``[0, 1]`` box yields exactly ``{0, 1}`` for EVERY relaxation value.

    This is the no-collateral property: the graduated binary enumeration (tanksize,
    5 free binaries -> 32 sub-NLPs) must cross the identical 2**k assignments after the
    generalization. Note ``x_rel == 1.0`` — the case that needs the ``ub - 1`` clamp on
    the floor, without which the pair would degenerate to ``{1}`` and halve the
    enumeration for every binary the relaxation pins at its upper bound.
    """
    assert set(S._native_seed_bracket(x_rel, 0.0, 1.0)) == {0.0, 1.0}


def test_bracket_on_general_box_tracks_the_relaxation():
    """On a wide box the pair brackets the relaxation instead of collapsing to 0/1.

    ``6.54`` on ``[0, 200]`` must give ``{6, 7}``. Before the fix this variable would be
    assigned 0 or 1 — off by two orders of magnitude on nvs-class boxes.
    """
    assert set(S._native_seed_bracket(6.54, 0.0, 200.0)) == {6.0, 7.0}
    assert set(S._native_seed_bracket(1.98, 0.0, 200.0)) == {1.0, 2.0}
    # At the box edges the pair stays inside the box.
    assert set(S._native_seed_bracket(200.0, 0.0, 200.0)) == {199.0, 200.0}
    assert set(S._native_seed_bracket(0.0, 0.0, 200.0)) == {0.0, 1.0}


def test_bracket_orders_nearest_first():
    """Nearest-first ordering. The enumeration is deadline-bounded and
    ``itertools.product`` is lexicographic, so on a wide box only a PREFIX of the 2**k
    product is reached — the first combination crossed must be the nearest-rounding
    point (the classic sub-NLP start), not an arbitrary corner."""
    assert S._native_seed_bracket(6.9, 0.0, 200.0)[0] == 7.0
    assert S._native_seed_bracket(6.1, 0.0, 200.0)[0] == 6.0


def test_bracket_never_emits_a_non_integer_or_out_of_box_value():
    """Candidates are assigned straight into integer slots, so a fractional endpoint
    would be a defect. A box narrower than one unit admits a single integer.

    The NON-INTEGER-ENDPOINT boxes matter: presolve can tighten a general integer to
    e.g. ``[0.5, 3.5]``, and clamping against the raw bounds rather than the integer box
    would hand back ``(2.5, 3.5)`` — two fractional values for an integer variable.
    """
    assert S._native_seed_bracket(0.3, 0.0, 0.6) == (0.0,)
    assert set(S._native_seed_bracket(3.4, 0.5, 3.5)) == {2.0, 3.0}
    checked = 0
    for lo_b, up_b in [
        (0.0, 1.0),
        (0.0, 200.0),
        (-5.0, 5.0),
        (3.0, 4.0),
        (0.0, 0.6),
        (0.5, 3.5),
        (-2.5, 2.5),
        (1.2, 1.8),  # no integer inside the box at all
    ]:
        import math

        box_has_an_integer = math.floor(up_b) >= math.ceil(lo_b)
        for x_rel in np.linspace(lo_b, up_b, 7):
            for v in S._native_seed_bracket(float(x_rel), lo_b, up_b):
                # Integrality is unconditional — a fractional value would land in an
                # integer slot.
                assert float(v).is_integer(), (lo_b, up_b, x_rel, v)
                checked += 1
                # Staying inside the box is required only when the box actually admits
                # an integer; ``[1.2, 1.8]`` admits none, and the documented behaviour
                # there is to return the nearest integer as a sub-NLP start point.
                if box_has_an_integer:
                    assert lo_b - 1e-12 <= v <= up_b + 1e-12, (lo_b, up_b, x_rel, v)
                    checked += 1
    assert checked > 0, "probe asserted nothing"


def test_bracket_handles_non_finite_relaxation_value():
    """A failed relaxation solve can leave a NaN in the base point; fall back to the box
    midpoint rather than propagating NaN into an integer slot."""
    out = S._native_seed_bracket(float("nan"), 0.0, 10.0)
    assert all(np.isfinite(v) and float(v).is_integer() for v in out)


# --------------------------------------------------------------------------------
# End-to-end: the seed on a general-integer model
# --------------------------------------------------------------------------------


def _wide_integer_model():
    """A general-integer model whose optimum is far from the ``{0,1}`` corner.

    Optimum 0 at ``(7, 12)``; the best point reachable with every variable pinned to 0
    or 1 is ``(1, 1)`` with objective ``36 + 121 = 157``. Synthetic and structural — the
    property under test is "the enumeration follows the box", which is a property of the
    variable class, not of any named instance (CLAUDE.md section 2).
    """
    m = Model()
    x = m.integer("x", lb=0, ub=20)
    y = m.integer("y", lb=0, ub=20)
    m.subject_to(x + y <= 30)
    m.minimize((x - 7) * (x - 7) + (y - 12) * (y - 12))
    return m


def test_seed_finds_the_optimum_of_a_wide_integer_model():
    """The regression test: before the fix the seed could only reach ``{0,1}^2`` and the
    best verified candidate was ~157; the bracket enumeration reaches ``(7, 12)``.

    ``sign=1.0, off=0.0`` is the identity objective mapping, so the returned internal
    value IS the model objective (the seed's documented contract is
    ``internal = sign * model_obj - off``).
    """
    m = _wide_integer_model()
    lb = np.array([0.0, 0.0])
    ub = np.array([20.0, 20.0])
    value, point = S._native_kernel_seed(m, lb, ub, 1.0, 0.0, 2)
    assert value is not None, "seed found no verified feasible point at all"
    assert value < 1.0, f"seed value {value} — the 0/1 corner best is ~157 (issue #902)"
    # The seed's point must itself be the genuinely-attained witness for that value.
    ok, obj = S._native_kernel_verify_point(m, point)
    assert ok is True
    assert abs(obj - value) < 1e-6


def test_seed_candidates_are_not_pinned_to_zero_one_on_a_wide_box():
    """Directly pins the mechanism, not just its outcome: at least one candidate must
    place a variable outside ``{0, 1}``. A seed that only ever emitted 0/1 points is the
    exact #902 signature."""
    import time

    m = _wide_integer_model()
    lb = np.array([0.0, 0.0])
    ub = np.array([20.0, 20.0])
    deadline = time.perf_counter() + 30.0
    seen = 0
    off_corner = 0
    for cand in S._native_kernel_seed_candidates(m, lb, ub, 2, deadline):
        seen += 1
        if np.any(np.asarray(cand)[:2] > 1.5):
            off_corner += 1
    assert seen > 0, "candidate generator yielded nothing — the probe asserted nothing"
    assert off_corner > 0, f"all {seen} candidates lay in {{0,1}}^2 (issue #902)"


def test_binary_model_seed_still_reaches_a_coupled_assignment():
    """No collateral on the binary class. ``x + y == 1`` with a strict preference for
    ``y`` means nearest-rounding the relaxation is not enough on its own — the
    enumeration has to cross the coupled assignment, exactly as it did for #764."""
    m = Model()
    x = m.binary("x")
    y = m.binary("y")
    m.subject_to(x + y == 1)
    m.minimize(3.0 * x - 5.0 * y)
    value, point = S._native_kernel_seed(m, np.array([0.0, 0.0]), np.array([1.0, 1.0]), 1.0, 0.0, 2)
    assert value is not None
    assert abs(value - (-5.0)) < 1e-6, f"expected the (0,1) assignment (-5.0), got {value}"
    assert np.allclose(point, [0.0, 1.0], atol=1e-6)


# --------------------------------------------------------------------------------
# The verification loop must not re-trace JAX per candidate
# --------------------------------------------------------------------------------


def test_verify_point_reuses_a_cached_evaluator(monkeypatch):
    """#902 profiling measured 697 JAX traces / 8.5 s inside a 12 s seed phase, because
    ``_native_kernel_verify_point`` built a fresh ``NLPEvaluator(model)`` per call — once
    per seed candidate. It now goes through ``cached_evaluator``.

    Asserted by construction count rather than by wall time: a timing assertion here
    would be a load-sensitive flake (CLAUDE.md section 9), while "how many evaluators
    did we build" is exactly the quantity that regressed.
    """
    from discopt._jax import nlp_evaluator as NE

    m = _wide_integer_model()
    x_ok = np.array([7.0, 12.0])

    builds = {"n": 0}
    real_init = NE.NLPEvaluator.__init__

    def counting_init(self, *a, **kw):
        builds["n"] += 1
        return real_init(self, *a, **kw)

    monkeypatch.setattr(NE.NLPEvaluator, "__init__", counting_init)

    ok, _ = S._native_kernel_verify_point(m, x_ok)
    assert ok is True
    after_first = builds["n"]
    assert after_first >= 1, "probe never built an evaluator — it asserted nothing"

    for _ in range(25):
        assert S._native_kernel_verify_point(m, x_ok)[0] is True
    assert builds["n"] == after_first, (
        f"{builds['n'] - after_first} evaluator rebuilds across 25 repeat verifications "
        "— the seed's verification loop is re-tracing JAX again (issue #902)"
    )
