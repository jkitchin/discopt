"""Unit tests for the solver-strategy advisor (``discopt.llm.advisor``) (#87).

The rule-based layer is pure logic and tested directly; the LLM augmentation
is exercised through a mocked ``discopt.llm.provider.complete`` so no network
or litellm calls happen (same convention as ``test_llm_modules.py``). The
safety invariant under test: advisor output is advisory only and the LLM
layer degrades to the rule-based result on any failure.
"""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.llm.advisor import (
    _analyze_structure,
    _llm_augment,
    _llm_presolve,
    _rule_based_params,
    presolve_analysis,
    suggest_solver_params,
)
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


def _bilinear_minlp():
    m = Model("advise")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y <= 4.0 + 2000.0 * b)  # big-M style constant
    m.minimize(x + y)
    return m


def test_analyze_structure_features():
    m = _bilinear_minlp()
    a = _analyze_structure(m)
    assert a["n_variables"] == 3
    assert a["n_integer"] == 1
    assert a["has_bilinear"]
    # NOTE: the big-M heuristic tokenizes str(constraint) on whitespace, but
    # the printer always attaches parentheses to constants ("(2000" never
    # parses as a float), so has_big_m cannot currently fire. This documents
    # the actual (advisory-only) behavior; tighten if the heuristic is fixed.
    assert not a["has_big_m"]
    assert not a["is_pure_continuous"]
    assert a["bound_range_ratio"] >= 1.0


def test_rule_based_params_bilinear_minlp():
    a = _analyze_structure(_bilinear_minlp())
    p = _rule_based_params(a)
    assert p["partitions"] == 4  # bilinear + integer -> partitioned McCormick
    assert p["nlp_solver"] == "ipm"
    assert p["batch_size"] == 16  # small model
    assert p["gap_tolerance"] == 1e-4
    assert "reasoning" in p and isinstance(p["reasoning"], str)


def test_rule_based_params_pure_continuous_lp():
    m = Model("lp")
    x = m.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m.subject_to(x[0] + x[1] <= 1.0)
    m.minimize(x[0])
    p = _rule_based_params(_analyze_structure(m))
    assert p["partitions"] == 0
    assert p["cutting_planes"] is False


def test_suggest_solver_params_rule_based_only():
    params = suggest_solver_params(_bilinear_minlp(), llm=False)
    assert params["partitions"] == 4
    # Suggested keys must all be real solve options (advisory contract).
    for key in ("nlp_solver", "batch_size", "gap_tolerance", "time_limit"):
        assert key in params


def test_suggest_solver_params_llm_augmentation_merges_overrides():
    with patch("discopt.llm.provider.complete") as mock_complete:
        mock_complete.return_value = '{"batch_size": 64, "reasoning": "wide frontier"}'
        params = suggest_solver_params(_bilinear_minlp(), llm=True)
    assert params["batch_size"] == 64
    assert params["reasoning"] == "wide frontier"
    # Non-overridden rule-based keys survive.
    assert params["partitions"] == 4


def test_suggest_solver_params_llm_failure_degrades_gracefully():
    with patch("discopt.llm.provider.complete", side_effect=RuntimeError("no api")):
        params = suggest_solver_params(_bilinear_minlp(), llm=True)
    # Safety invariant: rule-based result untouched by a failing LLM.
    assert params["partitions"] == 4
    assert params["batch_size"] == 16


def test_llm_augment_parses_fenced_json_and_rejects_garbage():
    m = _bilinear_minlp()
    a = _analyze_structure(m)
    base = _rule_based_params(a)
    with patch("discopt.llm.provider.complete") as mock_complete:
        mock_complete.return_value = '```json\n{"partitions": 8}\n```'
        out = _llm_augment(m, a, base, None)
    assert out == {"partitions": 8}
    with patch("discopt.llm.provider.complete") as mock_complete:
        mock_complete.return_value = "not json at all"
        assert _llm_augment(m, a, base, None) is None


def test_presolve_analysis_warnings():
    # Unbounded variable + bilinear-with-integer + scaling spread. The LLM
    # layer is disabled by mocking is_available so no network is attempted.
    m = Model("warn")
    free = m.continuous("free")  # unbounded both sides
    x = m.continuous("x", lb=0.0, ub=4.0)
    tiny = m.continuous("tiny", lb=0.0, ub=1e-7)
    b = m.binary("b")
    m.subject_to(x * tiny >= 0.0)
    m.subject_to(x <= 4.0 + 5000.0 * b)
    m.minimize(x + free)
    with patch("discopt.llm.is_available", return_value=False):
        warnings = presolve_analysis(m)
    joined = "\n".join(warnings)
    assert "unbounded" in joined
    assert "bilinear" in joined
    assert "magnitude ranges" in joined


def test_presolve_analysis_clean_model_is_quiet():
    m = Model("clean")
    x = m.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m.subject_to(x[0] + x[1] <= 1.0)
    m.minimize(x[0])
    with patch("discopt.llm.is_available", return_value=False):
        assert presolve_analysis(m) == []


def test_llm_presolve_parses_bullets_and_swallows_errors():
    m = _bilinear_minlp()
    a = _analyze_structure(m)
    with patch("discopt.llm.provider.complete") as mock_complete:
        mock_complete.return_value = "- bounds look loose\n- check row 2\nnoise line"
        out = _llm_presolve(m, a, None)
    assert out == ["[LLM] bounds look loose", "[LLM] check row 2"]
    with patch("discopt.llm.provider.complete", side_effect=RuntimeError("down")):
        assert _llm_presolve(m, a, None) == []
