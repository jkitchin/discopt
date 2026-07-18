"""Unit tests for the LLM tool-call ModelBuilder (#87).

The builder is the structured (``from_description``) path: every tool call
maps to a Model method through the sanitized safe-eval layer. Tests build a
real model tool-by-tool, then *solve it* and check the certified optimum —
the safety invariant being that LLM tool output can only produce a valid
Model, never touch solver math directly.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.llm.tools import ModelBuilder, execute_tool_calls

pytestmark = pytest.mark.unit


def _build_knapsack(builder: ModelBuilder):
    assert "Created model" in builder.execute_tool("create_model", {"name": "knap"})
    assert "binary" in builder.execute_tool("add_binary", {"name": "y", "shape": [3]})
    assert "parameter" in builder.execute_tool(
        "add_parameter", {"name": "v", "value": [6.0, 10.0, 12.0]}
    )
    assert "parameter" in builder.execute_tool(
        "add_parameter", {"name": "w", "value": [1.0, 2.0, 3.0]}
    )
    builder.execute_tool(
        "add_constraint",
        {"lhs": "sum(w * y)", "sense": "<=", "rhs": "4"},
    )
    builder.execute_tool("set_objective", {"expression": "sum(v * y)", "sense": "maximize"})


def test_model_builder_builds_and_solves_knapsack():
    b = ModelBuilder()
    _build_knapsack(b)
    assert b.model is not None
    res = b.model.solve(time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    # Capacity 4: best is items 2+3? w=5 too big; items 1+3 w=4 v=18;
    # items 1+2 w=3 v=16; item 2+... -> optimum 18.
    assert res.objective == pytest.approx(18.0, abs=1e-6)


def test_model_builder_variable_kinds_and_bounds():
    b = ModelBuilder()
    b.execute_tool("create_model", {"name": "kinds"})
    b.execute_tool("add_continuous", {"name": "x", "lb": -1.0, "ub": 2.0})
    b.execute_tool("add_integer", {"name": "i", "lb": 0, "ub": 5})
    var_map = {v.name: v for v in b.model._variables}
    assert float(np.asarray(var_map["x"].lb)) == -1.0
    assert float(np.asarray(var_map["x"].ub)) == 2.0
    assert float(np.asarray(var_map["i"].ub)) == 5.0


def test_model_builder_error_paths_are_messages_not_raises():
    b = ModelBuilder()
    # Tools before create_model return error strings (LLM-facing contract).
    assert b.execute_tool("add_binary", {"name": "y"}).startswith("Error")
    assert "Unknown tool" in b.execute_tool("definitely_not_a_tool", {})
    b.execute_tool("create_model", {"name": "m"})
    # A broken expression is reported, never raised.
    out = b.execute_tool(
        "add_constraint", {"lhs": "__import__('os').getcwd()", "sense": "<=", "rhs": "1"}
    )
    assert out.startswith("Error")


def test_execute_tool_calls_batch_interface():
    import json

    class _Call:
        def __init__(self, name, args):
            self.function = type("F", (), {"name": name, "arguments": json.dumps(args)})()
            self.id = f"call_{name}"

    calls = [
        _Call("create_model", {"name": "batch"}),
        _Call("add_continuous", {"name": "x", "lb": 0.0, "ub": 1.0}),
        _Call("set_objective", {"expression": "x", "sense": "minimize"}),
    ]
    b = ModelBuilder()
    results = execute_tool_calls(calls, b)
    assert len(results) == 3
    assert all(r["role"] == "tool" for r in results)
    assert b.model is not None and b.model._objective is not None
