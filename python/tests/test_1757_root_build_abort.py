"""#1757: the root-fallback BASE build honors its grant, not just its rows.

``root_build_deadline`` (#832) truncates the base ``build_milp_relaxation``'s
constraint rows at the fallback's grant, but the objective walk before them is not
interruptible by it. On a dense nonconvex quadratic objective that walk is nearly
the whole build: QPLIB_3709 at ``time_limit=30`` reached the #654 arm with a few
seconds left and its base build ran 36.7-40.1 s, so the solve returned at 57-60 s.
Measured directly (``_root_relaxation_lower_bound`` on 3709, 3.0 s grant): 39.93 s
and bound ``0.0`` flag-off, 3.08 s and ``None`` flag-on.

``DISCOPT_ROOT_BUILD_ABORT`` also passes ``abort_deadline`` = grant + one fallback
reserve (``_ROOT_BUILD_ABORT_GRACE_S``). These tests pin: past it the abort fires
and the fallback returns no bound (never a truncated one); a build that is late but
inside the grace keeps its bound; and a build that finishes in its grant is
unchanged by the flag.
"""

from __future__ import annotations

import time

import discopt._relax.milp_relaxation as MR
import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt.solver import _root_relaxation_lower_bound


def _dense_qp(n: int = 24, seed: int = 0) -> dm.Model:
    """A small dense nonconvex box QP: the QPLIB_3709 class in miniature."""
    rng = np.random.default_rng(seed)
    Q = rng.integers(-5, 6, size=(n, n)).astype(float)
    m = dm.Model("dense_qp")
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    m.minimize(sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(i + 1, n)))
    m.subject_to(sum(x) <= n / 2)
    return m


def _box(m):
    lb = np.array([v.lb for v in m._variables], dtype=np.float64).ravel()
    ub = np.array([v.ub for v in m._variables], dtype=np.float64).ravel()
    return lb, ub


def _record_aborts(monkeypatch):
    """Count base builds that raised ``RelaxationBuildAborted`` (§6: prove it fired)."""
    from discopt._relax.uniform_relax import RelaxationBuildAborted

    raised = []
    orig = MR.build_milp_relaxation

    def spy(*a, **k):
        try:
            return orig(*a, **k)
        except RelaxationBuildAborted:
            raised.append(k.get("abort_deadline"))
            raise

    monkeypatch.setattr(MR, "build_milp_relaxation", spy)
    return raised


def test_spent_grace_aborts_the_base_build_and_reports_no_bound(monkeypatch):
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_DEADLINE", "1")
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_ABORT", "1")
    monkeypatch.setattr(S, "_ROOT_BUILD_ABORT_GRACE_S", 0.0)
    raised = _record_aborts(monkeypatch)
    m = _dense_qp()
    lb, ub = _box(m)
    assert _root_relaxation_lower_bound(m, lb, ub, time_limit=0.0) is None
    assert len(raised) == 1 and raised[0] is not None


def test_late_build_inside_the_grace_keeps_its_bound(monkeypatch):
    """casctanks/hda at ``time_limit=5`` finish 0.1-0.2 s past a spent grant; an
    abort AT the grant threw those bounds away. Inside the grace they survive."""
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_DEADLINE", "1")
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_ABORT", "1")
    monkeypatch.setattr(S, "_ROOT_BUILD_ABORT_GRACE_S", 600.0)
    raised = _record_aborts(monkeypatch)
    m = _dense_qp()
    lb, ub = _box(m)
    b = _root_relaxation_lower_bound(m, lb, ub, time_limit=0.0)
    assert raised == []
    assert b is not None and np.isfinite(b)


def test_flag_off_keeps_the_whole_build(monkeypatch):
    """The opt-out control: a spent grant only truncates rows, the walk completes and
    the fully-linearized objective still yields a finite (weak) bound."""
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_DEADLINE", "1")
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_ABORT", "0")
    monkeypatch.setattr(S, "_ROOT_BUILD_ABORT_GRACE_S", 0.0)
    raised = _record_aborts(monkeypatch)
    m = _dense_qp()
    lb, ub = _box(m)
    b = _root_relaxation_lower_bound(m, lb, ub, time_limit=0.0)
    assert raised == []
    assert b is not None and np.isfinite(b)


@pytest.mark.parametrize("seed", [0, 1])
def test_unspent_grant_is_bound_identical(monkeypatch, seed):
    grant = 120.0
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_DEADLINE", "1")
    vals, walls = [], []
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_ROOT_BUILD_ABORT", flag)
        m = _dense_qp(seed=seed)
        lb, ub = _box(m)
        t = time.perf_counter()
        vals.append(_root_relaxation_lower_bound(m, lb, ub, time_limit=grant))
        walls.append(time.perf_counter() - t)
    # The premise (the grant did not bind) is asserted, not assumed (cf. #832's test).
    assert max(walls) < grant
    assert vals[0] is not None and vals[0] == vals[1]


def test_graduated_default_is_on(monkeypatch):
    """§5: graduated on introduction; ``=0`` stays the opt-out."""
    from discopt.solver_tuning import SolverTuning

    monkeypatch.delenv("DISCOPT_ROOT_BUILD_ABORT", raising=False)
    assert SolverTuning().root_build_abort is True
    monkeypatch.setenv("DISCOPT_ROOT_BUILD_ABORT", "0")
    assert SolverTuning().root_build_abort is False
