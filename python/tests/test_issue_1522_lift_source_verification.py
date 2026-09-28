"""The native kernel must verify its incumbent against the pre-reform model (#1522).

MINLPLib ``prob09`` (``min t`` s.t. ``100*(y - x**2)**2 + (1 - x)**2 - t == 0``) is
factorably lifted before the native spatial kernel sees it: the square is
distributed into ``100*y*y - 200*_fr_aux*y + 100*x**4`` with ``_fr_aux = x**2``. The
kernel's #789 check ran ``verify_point`` on that LIFT only. On the lifted row the
term magnitudes are ~450, where the source row's are ~2, and ``verify_point``'s
allowance ``ABS_TOL * max(1, term scale)`` grows with them. So a point with a
1.6e-4 residual on the source row was allowed 4.5e-4 on the lifted row. The kernel
reported it (``node_limit``, 100k nodes), and ``warm_start.check_feasibility``
rejected it at its default ``tol=1e-4``.

Before the fix, :func:`test_node_limit_incumbent_passes_check_feasibility` returned
the kernel's incumbent at obj ~0.011 with ``|e1| > 1e-4``.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import solver as S
from discopt import warm_start
from discopt._relax.factorable_reform import factorable_reformulate
from discopt.modeling.core import Model
from discopt.validation.feasibility import verify_point

pytestmark = pytest.mark.unit

# The incumbent the issue reported: obj 0.0033565, |e1| = 1.5976e-4.
_X, _Y = 1.03088, 1.05765
_E1_RESIDUAL = 1.5976e-4


def _prob09() -> Model:
    m = Model("prob09")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    t = m.continuous("t", lb=-100, ub=100)
    m.subject_to(100 * (y - x**2) ** 2 + (1 - x) ** 2 - t == 0, name="e1")
    m.minimize(t)
    return m


def _issue_point() -> tuple[np.ndarray, np.ndarray]:
    """``(source point, lifted point)``: the issue's point, with the aux exact."""
    f = 100 * (_Y - _X**2) ** 2 + (1 - _X) ** 2
    src = np.array([_X, _Y, f - _E1_RESIDUAL])
    return src, np.append(src, _X**2)


def _flat(result, names=("x", "y", "t")) -> np.ndarray:
    return np.array([float(result.x[n]) for n in names])


def test_lift_alone_accepts_a_point_the_source_rejects():
    """The mechanism itself: the lift's scale-keyed allowance covers the residual."""
    src_model = _prob09()
    lifted = factorable_reformulate(_prob09())
    assert [v.name for v in lifted._variables] == ["x", "y", "t", "_fr_aux_0"]
    src, lift = _issue_point()

    assert verify_point(lifted, lift).ok
    assert not verify_point(src_model, src).ok
    ok, viols = warm_start.check_feasibility(src_model, src)
    assert not ok and any("e1" in v for v in viols)


def test_kernel_verifier_rejects_it_when_given_the_source():
    lifted = factorable_reformulate(_prob09())
    src, lift = _issue_point()

    # Without a source this is the pre-#1522 check, and it accepts.
    assert S._native_kernel_verify_point(lifted, lift)[0] is True
    ok, obj = S._native_kernel_verify_point(lifted, lift, source=(_prob09(), 3))
    assert ok is False and obj is None
    assert S._native_kernel_source_only_failure(lifted, lift, (_prob09(), 3))

    # A genuinely feasible point still verifies against both, objective from the lift.
    good = np.array([1.0, 1.0, 0.0, 1.0])
    ok, obj = S._native_kernel_verify_point(lifted, good, source=(_prob09(), 3))
    assert ok is True and obj == pytest.approx(0.0, abs=1e-12)
    assert not S._native_kernel_source_only_failure(lifted, good, (_prob09(), 3))


def test_repair_returns_a_source_verified_point():
    lifted = factorable_reformulate(_prob09())
    _, lift = _issue_point()
    out = S._native_kernel_repair_point(lifted, lift, (_prob09(), 3), None)
    assert out is not None
    x_new, obj = out
    assert S._native_kernel_verify_point(lifted, x_new, source=(_prob09(), 3))[0]
    assert warm_start.check_feasibility(_prob09(), x_new[:3])[0]
    assert obj == pytest.approx(float(x_new[2]))


@pytest.mark.parametrize("max_nodes", [2000])
def test_node_limit_incumbent_passes_check_feasibility(monkeypatch, max_nodes):
    """The issue's exit shape: an unseeded kernel stopping at ``node_limit``.

    On the reporting machine the kernel's NLP seed found nothing and the tree ran to
    its node limit on its own McCormick incumbent. The seed is disabled here to reach
    that exit deterministically; everything downstream of it is the default path.
    """
    monkeypatch.setattr(S, "_native_kernel_seed", lambda *a, **k: (None, None))
    # The in-tree primal step (default ON) certifies this solve at node 1, so it
    # never reaches the repair exercised here; this is the ``=0`` path.
    monkeypatch.setenv("DISCOPT_NATIVE_NLP_PRIMAL", "0")
    m = _prob09()
    r = m.solve(deterministic=True, max_nodes=max_nodes)
    assert r.status == "node_limit"
    assert r.bound is not None and r.bound <= 1e-9
    assert r.x, "the repair found a verified point here; losing it is a regression"
    ok, viols = warm_start.check_feasibility(_prob09(), _flat(r))
    assert ok, viols
    assert verify_point(_prob09(), _flat(r)).ok
    assert r.objective >= r.bound - 1e-6


def test_time_limited_exit_never_reports_a_source_infeasible_point():
    """With no budget left to repair, the kernel keeps its bound and drops the point."""
    m = _prob09()
    r = m.solve(deterministic=True, time_limit=0.5)
    assert r.bound is not None
    if r.x:
        ok, viols = warm_start.check_feasibility(_prob09(), _flat(r))
        assert ok, viols
    else:
        assert r.objective is None


# --------------------------------------------------------------------------- #
# Item 4: why 100k nodes did not close, and the in-tree primal step
# --------------------------------------------------------------------------- #
# The dual side was never the problem: FBBT on the sum of squares gives ``t >= 0``,
# so the kernel's root bound is already the optimum. The kernel's only internal
# incumbents are LP vertices whose lifted terms are McCormick-tight, and on the
# curved valley ``y = x**2`` those appear only on tiny boxes, far from (1, 1): with
# no NLP seed, 100k nodes ended 2.3e-3 above the optimum. ``native_nlp_primal``
# lets the kernel run one verified local NLP from a node's LP point.

FLAG = "DISCOPT_NATIVE_NLP_PRIMAL"


def _unseeded_solve(monkeypatch, flag: str, max_nodes: int):
    monkeypatch.setattr(S, "_native_kernel_seed", lambda *a, **k: (None, None))
    monkeypatch.setenv(FLAG, flag)
    return _prob09().solve(deterministic=True, max_nodes=max_nodes)


def test_unseeded_kernel_cannot_close_without_the_primal_step(monkeypatch):
    r = _unseeded_solve(monkeypatch, "0", 2000)
    st = r.solver_stats or {}
    assert st.get("tree/nodes") == 2000.0, "must be the kernel's own node-limited exit"
    assert r.status == "node_limit"
    assert st.get("tree/primal_hook_calls") == 0.0


def test_primal_step_certifies_the_unseeded_kernel(monkeypatch):
    r = _unseeded_solve(monkeypatch, "1", 2000)
    st = r.solver_stats or {}
    assert st.get("tree/nodes") is not None, "must be the kernel's result, not a fallback"
    assert st["tree/primal_hook_calls"] >= 1 and st["tree/primal_hook_improvements"] >= 1
    assert r.status == "optimal" and r.gap_certified
    assert st["tree/nodes"] < 2000
    assert r.bound <= 1e-9 and r.objective == pytest.approx(0.0, abs=1e-6)
    ok, viols = warm_start.check_feasibility(_prob09(), _flat(r))
    assert ok, viols


def test_primal_hook_returns_only_verified_improvements(monkeypatch):
    lifted = factorable_reformulate(_prob09())
    src, lift = _issue_point()
    hook = S._native_kernel_primal_hook(lifted, (_prob09(), 3), 1.0, 0.0, 4, None)

    got = hook(np.append(lift, [7.0, 7.0]), None)  # trailing McCormick columns ignored
    assert got is not None
    value, point = got
    assert len(point) == 4
    assert S._native_kernel_verify_point(lifted, np.array(point), source=(_prob09(), 3))[0]
    assert value == pytest.approx(point[2])
    # Not an improvement on an incumbent it cannot beat.
    assert hook(lift, value - 1.0) is None
    # Nothing is returned when the local NLP's point does not verify.
    monkeypatch.setattr(S, "_native_kernel_verify_point", lambda *a, **k: (False, None))
    assert hook(lift, None) is None
    assert hook.counts["calls"] == 3 and hook.counts["verified"] == 1


def test_a_primal_hook_defect_is_raised_not_skipped(monkeypatch):
    """The kernel call sits under a defensive ``except`` that logs "skipped" at
    DEBUG; an exception from OUR hook must not be downgraded to that (CLAUDE.md §7).
    """

    def boom(*a, **k):
        raise RuntimeError("hook defect #1522")

    monkeypatch.setattr(S, "_native_kernel_repair_point", boom)
    with pytest.raises(RuntimeError, match="hook defect #1522"):
        _unseeded_solve(monkeypatch, "1", 50)


def _all_integer_model() -> Model:
    m = Model("int_product")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.subject_to(x * y >= 2, name="prod")
    m.minimize(x + y)
    return m


def test_all_integer_step_verifies_the_rounded_point_without_an_nlp(monkeypatch):
    """With every column fixed, the local NLP is a solve over one point.

    Measured under the in-tree hook on nvs13 / nvs18 / nvs17: the POUNCE solve
    returned exactly the rounded point with the same verdict on 39 of 39 calls, at
    ~40x the cost of verifying it directly. So the step verifies it directly.
    """

    def no_nlp(*a, **k):
        raise AssertionError("an all-integer point must not reach the NLP solver")

    monkeypatch.setattr(S, "_solve_node_nlp_kkt", no_nlp)
    m = _all_integer_model()

    out = S._native_kernel_repair_point(m, np.array([1.2, 1.9]), None, None)
    assert out is not None
    x_new, obj = out
    assert np.array_equal(x_new, [1.0, 2.0]) and obj == pytest.approx(3.0)
    # The rounded point (0, 0) violates x*y >= 2: nothing is returned.
    assert S._native_kernel_repair_point(m, np.array([0.4, 0.4]), None, None) is None
