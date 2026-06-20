"""Tests for :class:`discopt.SolverTuning` — the typed replacement for the
algorithmic ``DISCOPT_*`` tuning environment variables.

Covers three properties that the old scattered ``os.environ.get`` reads lacked:
per-call resolution (env is a *default*, resolved at construction, never frozen),
explicit-field-wins-over-env, and up-front validation. A functional test proves
the relaxer read sites actually consult the published tuning rather than the
environment.
"""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import discopt.modeling as dm  # noqa: E402
from discopt import SolverTuning  # noqa: E402
from discopt.solver_tuning import current, reset_current, set_current  # noqa: E402

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# env-default resolution (the de-freeze) + explicit override
# --------------------------------------------------------------------------- #
def test_defaults_match_legacy_env_defaults():
    t = SolverTuning()
    assert t.rlt_quad is True
    assert t.rlt_quad_max == 256
    assert t.multilinear_rlt_max == 4
    assert t.node_bound_mode == "lp"
    assert t.node_nlp_stride == 4
    assert t.lp_warmstart is True
    assert t.rlt is False
    assert t.lifted_fbbt is False
    assert t.trilinear_nested is False


@pytest.mark.parametrize(
    "env_name, env_val, field, expected",
    [
        ("DISCOPT_RLT_QUAD", "0", "rlt_quad", False),
        ("DISCOPT_RLT", "1", "rlt", True),
        ("DISCOPT_LIFTED_FBBT", "1", "lifted_fbbt", True),
        ("DISCOPT_NODE_BOUND_MODE", "milp", "node_bound_mode", "milp"),
        ("DISCOPT_NODE_NLP_STRIDE", "2", "node_nlp_stride", 2),
        ("DISCOPT_RLT_QUAD_MAX", "64", "rlt_quad_max", 64),
        ("DISCOPT_TRILINEAR", "nested", "trilinear_nested", True),
        ("DISCOPT_SQUARE_SEPARATE", "0", "square_separate", False),
        ("DISCOPT_LP_WARMSTART", "0", "lp_warmstart", False),
    ],
)
def test_env_is_resolved_at_construction_not_import(
    monkeypatch, env_name, env_val, field, expected
):
    # setenv happens *after* discopt is imported; the default_factory reads it at
    # construction, so the value is picked up (the old module-frozen reads could not).
    monkeypatch.setenv(env_name, env_val)
    assert getattr(SolverTuning(), field) == expected


def test_explicit_field_overrides_env(monkeypatch):
    monkeypatch.setenv("DISCOPT_RLT_QUAD", "0")
    monkeypatch.setenv("DISCOPT_NODE_BOUND_MODE", "milp")
    t = SolverTuning(rlt_quad=True, node_bound_mode="lp")
    assert t.rlt_quad is True  # explicit beats env
    assert t.node_bound_mode == "lp"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(rlt_quad_max=0), "rlt_quad_max must be >= 1"),
        (dict(multilinear_rlt_max=0), "multilinear_rlt_max must be >= 1"),
        (dict(node_nlp_stride=0), "node_nlp_stride must be >= 1"),
        (dict(node_bound_mode="bogus"), "node_bound_mode must be 'lp' or 'milp'"),
    ],
)
def test_validation_rejects_out_of_range(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SolverTuning(**kwargs)


def test_replace_validates_and_rejects_unknown_field():
    t = SolverTuning()
    assert t.replace(rlt_quad=False).rlt_quad is False
    with pytest.raises(ValueError, match="node_nlp_stride"):
        t.replace(node_nlp_stride=-1)
    with pytest.raises(TypeError, match="unknown SolverTuning field"):
        t.replace(not_a_field=1)


def test_is_frozen():
    t = SolverTuning()
    with pytest.raises(Exception):  # dataclass(frozen=True) -> FrozenInstanceError
        t.rlt_quad = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# context var publish/reset
# --------------------------------------------------------------------------- #
def test_current_outside_scope_is_env_default(monkeypatch):
    monkeypatch.setenv("DISCOPT_NODE_NLP_STRIDE", "7")
    assert current().node_nlp_stride == 7  # fresh env-resolved, no scope active


def test_set_and_reset_current_round_trip():
    token = set_current(SolverTuning(node_nlp_stride=9))
    try:
        assert current().node_nlp_stride == 9
    finally:
        reset_current(token)
    # back to env default (4) once the scope is reset
    assert current().node_nlp_stride == 4


# --------------------------------------------------------------------------- #
# functional: the relaxer read sites consult current(), not os.environ
# --------------------------------------------------------------------------- #
def test_relaxer_reads_published_tuning():
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    m = dm.Model("r")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(x * y)
    m.subject_to(x + y <= 1.0)

    token = set_current(SolverTuning(node_bound_mode="milp", lifted_fbbt=True))
    try:
        relaxer = MccormickLPRelaxer(m)
        assert relaxer._lp_node_bound is False  # node_bound_mode="milp" -> not LP
        assert relaxer._lifted_fbbt is True
    finally:
        reset_current(token)


def test_solve_accepts_tuning_object():
    m = dm.Model("s")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x * y + x * x)
    m.subject_to(x + y >= -1.0)
    r = m.solve(time_limit=10.0, tuning=SolverTuning(rlt_quad=False, node_nlp_stride=2))
    assert r.objective is not None


def test_solve_does_not_leak_tuning_to_later_relaxer():
    """A solve's tuning override is scoped to that call — it must not linger and
    pollute a relaxer built afterwards outside any solve (regression: the publish
    used to never reset)."""
    m = dm.Model("leak")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x * y + x * x)
    m.subject_to(x + y >= -1.0)
    m.solve(time_limit=10.0, tuning=SolverTuning(node_bound_mode="milp", lifted_fbbt=True))
    # Back to env defaults once the solve returns.
    assert current().node_bound_mode == "lp"
    assert current().lifted_fbbt is False
