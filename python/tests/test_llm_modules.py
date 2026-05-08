"""Tests for discopt.llm submodules.

Tests safety, serializer, prompts, tools, diagnosis, reformulation,
commentary, chat, and provider modules. All tests mock LLM calls
so no network access is needed.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import SolveResult

# ─────────────────────────────────────────────────────────────
# Helper fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_model():
    m = dm.Model("test_model")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    y = m.binary("y", shape=(2,))
    m.minimize(dm.sum(x) + 5 * dm.sum(y))
    m.subject_to(x[0] + x[1] >= 3, name="min_flow")
    m.subject_to(dm.sum(y) <= 1, name="max_active")
    return m


@pytest.fixture()
def optimal_result():
    return SolveResult(
        status="optimal",
        objective=12.5,
        bound=12.5,
        gap=0.0,
        x={"x": np.array([1.5, 1.5, 0.0]), "y": np.array([1.0, 0.0])},
        wall_time=0.5,
        node_count=10,
        rust_time=0.1,
        jax_time=0.3,
        python_time=0.1,
    )


# ─────────────────────────────────────────────────────────────
# safety.py
# ─────────────────────────────────────────────────────────────


class TestSafetyValidateExplanation:
    def test_normal_text_unchanged(self):
        from discopt.llm.safety import validate_explanation

        text = "The model solved optimally."
        assert validate_explanation(text) == text

    def test_empty_string(self):
        from discopt.llm.safety import validate_explanation

        assert validate_explanation("") == "No explanation available."

    def test_whitespace_only(self):
        from discopt.llm.safety import validate_explanation

        assert validate_explanation("   \n  ") == "No explanation available."

    def test_none_input(self):
        from discopt.llm.safety import validate_explanation

        assert validate_explanation(None) == "No explanation available."

    def test_truncation_at_5000_chars(self):
        from discopt.llm.safety import validate_explanation

        text = "a" * 6000
        result = validate_explanation(text)
        assert len(result) <= 5000 + len("\n\n[Explanation truncated]")
        assert result.endswith("[Explanation truncated]")

    def test_strips_whitespace(self):
        from discopt.llm.safety import validate_explanation

        assert validate_explanation("  hello  ") == "hello"


class TestSafetyValidateModel:
    def test_valid_model(self, simple_model):
        from discopt.llm.safety import validate_model

        warnings = validate_model(simple_model)
        assert isinstance(warnings, list)

    def test_no_objective_warns(self):
        from discopt.llm.safety import validate_model

        m = dm.Model("no_obj")
        m.continuous("x", lb=0, ub=10)
        warnings = validate_model(m)
        assert any("Validation error" in w for w in warnings)

    def test_no_constraints_warns(self):
        from discopt.llm.safety import validate_model

        m = dm.Model("no_con")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        warnings = validate_model(m)
        assert any("no constraints" in w for w in warnings)

    def test_many_variables_warns(self):
        from discopt.llm.safety import validate_model

        m = dm.Model("big")
        m.continuous("x", shape=(60,), lb=0, ub=1)
        m.minimize(0)
        m.subject_to(m._variables[0][0] >= 0, name="c")
        warnings = validate_model(m)
        assert any("over-generate" in w for w in warnings)


class TestSafetySanitizeToolArgs:
    def test_name_sanitization(self):
        from discopt.llm.safety import sanitize_tool_args

        result = sanitize_tool_args("add_continuous", {"name": "my var!@#$"})
        assert result["name"] == "my_var____"

    def test_continuous_bounds_clamped(self):
        from discopt.llm.safety import sanitize_tool_args

        result = sanitize_tool_args("add_continuous", {"name": "x", "lb": -1e30, "ub": 1e30})
        assert result["lb"] == -1e20
        assert result["ub"] == 1e20

    def test_continuous_shape_validation(self):
        from discopt.llm.safety import sanitize_tool_args

        with pytest.raises(ValueError, match="Invalid dimension"):
            sanitize_tool_args("add_continuous", {"name": "x", "shape": [0]})

        with pytest.raises(ValueError, match="Invalid dimension"):
            sanitize_tool_args("add_continuous", {"name": "x", "shape": [100000]})

    def test_integer_bounds_cast(self):
        from discopt.llm.safety import sanitize_tool_args

        result = sanitize_tool_args("add_integer", {"name": "n", "lb": 1.5, "ub": 10.9})
        assert result["lb"] == 1
        assert result["ub"] == 10

    def test_set_objective_valid_sense(self):
        from discopt.llm.safety import sanitize_tool_args

        result = sanitize_tool_args("set_objective", {"sense": "minimize", "expression": "x"})
        assert result["sense"] == "minimize"

    def test_set_objective_invalid_sense(self):
        from discopt.llm.safety import sanitize_tool_args

        with pytest.raises(ValueError, match="Invalid objective sense"):
            sanitize_tool_args("set_objective", {"sense": "minmax", "expression": "x"})


# ─────────────────────────────────────────────────────────────
# serializer.py
# ─────────────────────────────────────────────────────────────


class TestSerializeModel:
    def test_contains_model_name(self, simple_model):
        from discopt.llm.serializer import serialize_model

        text = serialize_model(simple_model)
        assert "test_model" in text

    def test_contains_variables(self, simple_model):
        from discopt.llm.serializer import serialize_model

        text = serialize_model(simple_model)
        assert "x" in text
        assert "y" in text
        assert "continuous" in text
        assert "binary" in text

    def test_contains_constraints(self, simple_model):
        from discopt.llm.serializer import serialize_model

        text = serialize_model(simple_model)
        assert "Constraints" in text
        assert "min_flow" in text

    def test_contains_statistics(self, simple_model):
        from discopt.llm.serializer import serialize_model

        text = serialize_model(simple_model)
        assert "Statistics" in text
        assert "Total variables" in text


class TestSerializeSolveResult:
    def test_contains_status_and_objective(self, simple_model, optimal_result):
        from discopt.llm.serializer import serialize_solve_result

        text = serialize_solve_result(optimal_result, model=simple_model)
        assert "optimal" in text
        assert "12.5" in text

    def test_contains_timing(self, optimal_result):
        from discopt.llm.serializer import serialize_solve_result

        text = serialize_solve_result(optimal_result)
        assert "Wall time" in text
        assert "Rust" in text
        assert "JAX" in text

    def test_contains_solution(self, optimal_result):
        from discopt.llm.serializer import serialize_solve_result

        text = serialize_solve_result(optimal_result)
        assert "Solution" in text

    def test_no_solution(self):
        from discopt.llm.serializer import serialize_solve_result

        result = SolveResult(status="infeasible", wall_time=0.1)
        text = serialize_solve_result(result)
        assert "infeasible" in text
        assert "Solution" not in text


class TestSerializeDataSchema:
    def test_numpy_array(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {"costs": np.array([1.0, 2.0, 3.0])}
        text = serialize_data_schema(data)
        assert "costs" in text
        assert "numpy" in text.lower() or "array" in text.lower()
        assert "Sample values" in text

    def test_dict_data(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {"config": {"a": 1, "b": 2, "c": 3}}
        text = serialize_data_schema(data)
        assert "config" in text
        assert "dict" in text
        assert "3 keys" in text

    def test_list_data(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {"items": [10, 20, 30]}
        text = serialize_data_schema(data)
        assert "items" in text
        assert "3 elements" in text

    def test_scalar_data(self):
        from discopt.llm.serializer import serialize_data_schema

        data = {"threshold": 0.5}
        text = serialize_data_schema(data)
        assert "threshold" in text
        assert "float" in text


class TestSerializerHelpers:
    def test_format_bound_scalar(self):
        from discopt.llm.serializer import _format_bound

        assert _format_bound(5.0, "lb") == "lb=5"
        assert _format_bound(-1e20, "lb") == "lb=-inf"
        assert _format_bound(1e20, "ub") == "ub=+inf"

    def test_format_bound_array(self):
        from discopt.llm.serializer import _format_bound

        assert "lb=" in _format_bound(np.array([1.0]), "lb")
        assert "array" in _format_bound(np.array([1.0, 2.0]), "lb")

    def test_format_value_small_array(self):
        from discopt.llm.serializer import _format_value

        result = _format_value(np.array([1, 2, 3]))
        assert "1" in result

    def test_format_value_large_array(self):
        from discopt.llm.serializer import _format_value

        result = _format_value(np.arange(20))
        assert "array" in result
        assert "mean" in result

    def test_pct_normal(self):
        from discopt.llm.serializer import _pct

        assert _pct(25, 100) == "25.0%"

    def test_pct_zero_total(self):
        from discopt.llm.serializer import _pct

        assert _pct(10, 0) == "N/A"


# ─────────────────────────────────────────────────────────────
# prompts.py
# ─────────────────────────────────────────────────────────────


class TestPrompts:
    def test_get_explain_prompt_all_statuses(self):
        from discopt.llm.prompts import get_explain_prompt

        for status in [
            "optimal",
            "feasible",
            "infeasible",
            "iteration_limit",
            "time_limit",
            "node_limit",
            "unknown_status",
        ]:
            prompt = get_explain_prompt(status)
            assert isinstance(prompt, str)
            assert "{model_text}" in prompt
            assert "{result_text}" in prompt

    def test_optimal_prompt_content(self):
        from discopt.llm.prompts import EXPLAIN_OPTIMAL

        assert "optimal" in EXPLAIN_OPTIMAL.lower()
        assert "{model_text}" in EXPLAIN_OPTIMAL

    def test_infeasible_prompt_content(self):
        from discopt.llm.prompts import EXPLAIN_INFEASIBLE

        assert "infeasible" in EXPLAIN_INFEASIBLE.lower()
        assert "relax" in EXPLAIN_INFEASIBLE.lower()

    def test_formulate_prompts(self):
        from discopt.llm.prompts import FORMULATE_SYSTEM, FORMULATE_USER

        assert "discopt" in FORMULATE_SYSTEM.lower()
        assert "{description}" in FORMULATE_USER
        assert "{data_schema}" in FORMULATE_USER

    def test_teaching_prompts(self):
        from discopt.llm.prompts import TEACHING_SYSTEM, TEACHING_WALKTHROUGH

        assert "teaching" in TEACHING_SYSTEM.lower()
        assert "{model_text}" in TEACHING_WALKTHROUGH
        assert "{result_text}" in TEACHING_WALKTHROUGH

    def test_debug_prompts(self):
        from discopt.llm.prompts import DEBUG_QUESTION, DEBUG_SYSTEM

        assert "debug" in DEBUG_SYSTEM.lower()
        assert "{question}" in DEBUG_QUESTION
        assert "{model_text}" in DEBUG_QUESTION
        assert "{result_text}" in DEBUG_QUESTION

    def test_explain_system_prompt(self):
        from discopt.llm.prompts import EXPLAIN_SYSTEM

        assert "optimization" in EXPLAIN_SYSTEM.lower()


# ─────────────────────────────────────────────────────────────
# tools.py
# ─────────────────────────────────────────────────────────────


class TestToolDefinitions:
    def test_tool_definitions_structure(self):
        from discopt.llm.tools import TOOL_DEFINITIONS

        assert isinstance(TOOL_DEFINITIONS, list)
        assert len(TOOL_DEFINITIONS) > 0

        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_expected_tools_present(self):
        from discopt.llm.tools import TOOL_DEFINITIONS

        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "create_model",
            "add_continuous",
            "add_binary",
            "add_integer",
            "add_parameter",
            "set_objective",
            "add_constraint",
            "add_if_then",
            "add_either_or",
            "add_implies",
            "add_at_least",
        }
        assert expected.issubset(names)


class TestModelBuilder:
    def test_create_model(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        result = builder.execute_tool("create_model", {"name": "test"})
        assert "Created" in result
        assert builder.model is not None
        assert builder.model.name == "test"

    def test_add_continuous(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        result = builder.execute_tool(
            "add_continuous", {"name": "x", "shape": [3], "lb": 0, "ub": 10}
        )
        assert "Added" in result
        assert builder.model.num_variables == 3
        assert "x" in builder._namespace

    def test_add_binary(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        result = builder.execute_tool("add_binary", {"name": "y", "shape": [2]})
        assert "Added" in result
        assert builder.model.num_integer == 2

    def test_add_integer(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        result = builder.execute_tool("add_integer", {"name": "n", "shape": [], "lb": 0, "ub": 5})
        assert "Added" in result

    def test_add_parameter(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        result = builder.execute_tool("add_parameter", {"name": "c", "value": 3.14})
        assert "Added parameter" in result
        assert "c" in builder._namespace

    def test_add_parameter_array(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        result = builder.execute_tool("add_parameter", {"name": "v", "value": [1, 2, 3]})
        assert "Added parameter" in result

    def test_set_objective_minimize(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "set_objective", {"sense": "minimize", "expression": "x[0] + x[1]"}
        )
        assert "Set objective" in result

    def test_set_objective_maximize(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "set_objective", {"sense": "maximize", "expression": "x[0] + x[1]"}
        )
        assert "Set objective" in result

    def test_add_constraint(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "add_constraint",
            {"lhs": "x[0] + x[1]", "sense": ">=", "rhs": "5", "name": "min_sum"},
        )
        assert "Added constraint" in result
        assert builder.model.num_constraints == 1

    def test_add_constraint_le(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "add_constraint",
            {"lhs": "x[0]", "sense": "<=", "rhs": "5", "name": "upper"},
        )
        assert "Added constraint" in result

    def test_add_constraint_eq(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [2], "lb": 0, "ub": 10})
        result = builder.execute_tool(
            "add_constraint",
            {"lhs": "x[0]", "sense": "==", "rhs": "x[1]", "name": "equal"},
        )
        assert "Added constraint" in result

    def test_unknown_tool(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        result = builder.execute_tool("nonexistent", {})
        assert "Unknown tool" in result

    def test_tool_before_model_creation(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        result = builder.execute_tool("add_continuous", {"name": "x"})
        assert "Error" in result

    def test_eval_expression_invalid(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})
        result = builder.execute_tool(
            "set_objective", {"sense": "minimize", "expression": "undefined_var"}
        )
        assert "Error" in result

    def test_full_model_build_and_validate(self):
        from discopt.llm.tools import ModelBuilder

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "transport"})
        builder.execute_tool("add_continuous", {"name": "x", "shape": [3], "lb": 0, "ub": 100})
        builder.execute_tool(
            "set_objective", {"sense": "minimize", "expression": "x[0] + 2*x[1] + 3*x[2]"}
        )
        builder.execute_tool(
            "add_constraint",
            {"lhs": "x[0] + x[1] + x[2]", "sense": ">=", "rhs": "50", "name": "demand"},
        )
        model = builder.model
        assert model.name == "transport"
        assert model.num_variables == 3
        assert model.num_constraints == 1
        model.validate()


class TestExecuteToolCalls:
    def test_execute_batch(self):
        from discopt.llm.tools import ModelBuilder, execute_tool_calls

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})

        @dataclass
        class FakeFunction:
            name: str
            arguments: str

        @dataclass
        class FakeCall:
            id: str
            function: FakeFunction

        calls = [
            FakeCall(
                id="call_1",
                function=FakeFunction(
                    name="add_continuous",
                    arguments='{"name": "x", "shape": [2], "lb": 0, "ub": 10}',
                ),
            ),
        ]
        results = execute_tool_calls(calls, builder)
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert "Added" in results[0]["content"]

    def test_execute_invalid_json(self):
        from discopt.llm.tools import ModelBuilder, execute_tool_calls

        builder = ModelBuilder()
        builder.execute_tool("create_model", {"name": "t"})

        @dataclass
        class FakeFunction:
            name: str
            arguments: str

        @dataclass
        class FakeCall:
            id: str
            function: FakeFunction

        calls = [
            FakeCall(id="call_bad", function=FakeFunction(name="add_continuous", arguments="{")),
        ]
        results = execute_tool_calls(calls, builder)
        assert "Error parsing" in results[0]["content"]


# ─────────────────────────────────────────────────────────────
# diagnosis.py
# ─────────────────────────────────────────────────────────────


class TestDiagnosisInfeasibility:
    @pytest.mark.slow
    def test_diagnose_infeasibility_basic(self):
        from discopt.llm.diagnosis import diagnose_infeasibility

        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 20, name="impossible")

        result = SolveResult(status="infeasible", wall_time=0.1)
        report = diagnose_infeasibility(m, result)
        assert isinstance(report, str)
        assert "infeasib" in report.lower()

    def test_diagnose_infeasibility_with_mock_llm(self):
        from discopt.llm.diagnosis import diagnose_infeasibility

        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 20, name="impossible")

        result = SolveResult(
            status="infeasible",
            wall_time=0.1,
            x={"x": np.array([10.0])},
        )

        with (
            patch("discopt.llm.is_available", return_value=True),
            patch("discopt.llm.diagnosis._llm_diagnose", return_value="LLM says conflict"),
        ):
            report = diagnose_infeasibility(m, result)
            assert report == "LLM says conflict"

    def test_diagnose_infeasibility_llm_fails_gracefully(self):
        from discopt.llm.diagnosis import diagnose_infeasibility

        m = dm.Model("infeasible")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x >= 20, name="impossible")

        result = SolveResult(status="infeasible", wall_time=0.1)

        with patch("discopt.llm.is_available", side_effect=Exception("fail")):
            report = diagnose_infeasibility(m, result)
            assert "infeasib" in report.lower()


class TestDiagnosisAnalyzeBounds:
    def test_empty_domain_detected(self):
        from discopt.llm.diagnosis import _analyze_bounds

        m = dm.Model("bad_bounds")
        m.continuous("x", lb=10, ub=5)
        m.minimize(0)
        issues = _analyze_bounds(m)
        assert any(i["issue"] == "empty_domain" for i in issues)

    def test_near_fixed_detected(self):
        from discopt.llm.diagnosis import _analyze_bounds

        m = dm.Model("tight")
        m.continuous("x", lb=5.0, ub=5.0 + 1e-8)
        m.minimize(0)
        issues = _analyze_bounds(m)
        assert any(i["issue"] == "near_fixed" for i in issues)


class TestDiagnosisResult:
    def test_dispatch_time_limit(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)

        result = SolveResult(
            status="time_limit",
            wall_time=100.0,
            node_count=500,
            gap=0.05,
            rust_time=30.0,
            jax_time=60.0,
            python_time=10.0,
        )
        report = diagnose_result(m, result)
        assert "time" in report.lower()

    def test_dispatch_node_limit(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)

        result = SolveResult(
            status="node_limit",
            wall_time=50.0,
            node_count=10000,
            gap=0.10,
            rust_time=20.0,
            jax_time=25.0,
            python_time=5.0,
        )
        report = diagnose_result(m, result)
        assert "node" in report.lower()

    def test_dispatch_iteration_limit(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        result = SolveResult(status="iteration_limit", wall_time=1.0)
        report = diagnose_result(m, result)
        assert "iteration" in report.lower()
        assert "max_iter" in report

    def test_dispatch_optimal_no_issues(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        result = SolveResult(status="optimal", gap=0.0, wall_time=0.1)
        report = diagnose_result(m, result)
        assert "No issues" in report

    def test_dispatch_large_gap(self):
        from discopt.llm.diagnosis import diagnose_result

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        result = SolveResult(
            status="optimal",
            gap=0.15,
            wall_time=10.0,
            objective=100.0,
            bound=85.0,
        )
        report = diagnose_result(m, result)
        assert "gap" in report.lower()

    def test_time_limit_gap_categories(self):
        from discopt.llm.diagnosis import _diagnose_limit

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)

        # Small gap
        r1 = SolveResult(
            status="time_limit",
            wall_time=10.0,
            node_count=100,
            gap=0.005,
            rust_time=3.0,
            jax_time=5.0,
            python_time=2.0,
        )
        report1 = _diagnose_limit(m, r1, None)
        assert "near-optimal" in report1

        # Moderate gap
        r2 = SolveResult(
            status="time_limit",
            wall_time=10.0,
            node_count=100,
            gap=0.05,
            rust_time=3.0,
            jax_time=5.0,
            python_time=2.0,
        )
        report2 = _diagnose_limit(m, r2, None)
        assert "partitions" in report2.lower()

        # Large gap
        r3 = SolveResult(
            status="time_limit",
            wall_time=10.0,
            node_count=100,
            gap=0.20,
            rust_time=3.0,
            jax_time=5.0,
            python_time=2.0,
        )
        report3 = _diagnose_limit(m, r3, None)
        assert "weak" in report3.lower()


# ─────────────────────────────────────────────────────────────
# reformulation.py
# ─────────────────────────────────────────────────────────────


class TestReformulationSuggestion:
    def test_dataclass_fields(self):
        from discopt.llm.reformulation import ReformulationSuggestion

        s = ReformulationSuggestion(
            category="big_m_tightening",
            description="Tighten M",
            impact="Faster solve",
            auto_applicable=True,
            details={"M_value": 1000},
        )
        assert s.category == "big_m_tightening"
        assert s.auto_applicable is True
        assert s.details["M_value"] == 1000


class TestReformulationDetection:
    def test_detect_weak_bounds(self):
        from discopt.llm.reformulation import analyze_reformulations

        m = dm.Model("t")
        m.continuous("x", shape=(3,))  # unbounded
        m.binary("y")
        m.minimize(0)
        m.subject_to(m._variables[1] >= 0, name="c")

        suggestions = analyze_reformulations(m)
        assert any(s.category == "bound_tightening" for s in suggestions)

    def test_detect_bilinear(self):
        from discopt.llm.reformulation import analyze_reformulations

        m = dm.Model("bilinear")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        m.subject_to(x * y <= 50, name="bilinear_con")

        suggestions = analyze_reformulations(m)
        categories = [s.category for s in suggestions]
        assert "mccormick_partitioning" in categories

    def test_detect_symmetry(self):
        from discopt.llm.reformulation import _detect_symmetry

        m = dm.Model("symmetric")
        m.binary("y1", shape=(5,))
        m.binary("y2", shape=(5,))
        m.minimize(0)

        suggestions = _detect_symmetry(m)
        assert any(s.category == "symmetry_breaking" for s in suggestions)

    def test_no_symmetry_different_sizes(self):
        from discopt.llm.reformulation import _detect_symmetry

        m = dm.Model("asym")
        m.binary("y1", shape=(3,))
        m.binary("y2", shape=(5,))
        m.minimize(0)

        suggestions = _detect_symmetry(m)
        assert len(suggestions) == 0

    def test_detect_big_m_returns_list(self):
        from discopt.llm.reformulation import _detect_big_m

        m = dm.Model("bigm")
        x = m.continuous("x", lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x)
        m.subject_to(x <= 10000 * y, name="big_m_con")

        suggestions = _detect_big_m(m)
        # _detect_big_m does heuristic token parsing; returns a list
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert s.category == "big_m_tightening"

    def test_no_big_m_without_binary(self):
        from discopt.llm.reformulation import _detect_big_m

        m = dm.Model("no_binary")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x <= 5, name="c")

        suggestions = _detect_big_m(m)
        assert len(suggestions) == 0

    def test_apply_bound_tightening(self):
        from discopt.llm.reformulation import apply_bound_tightening

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=100)
        m.minimize(0)
        count = apply_bound_tightening(m)
        assert isinstance(count, int)

    def test_analyze_with_llm_mocked(self):
        from discopt.llm.reformulation import analyze_reformulations

        m = dm.Model("t")
        m.continuous("x", lb=0, ub=10)
        m.minimize(0)
        m.subject_to(m._variables[0] >= 0, name="c")

        with (
            patch("discopt.llm.is_available", return_value=True),
            patch("discopt.llm.reformulation._llm_analyze", return_value=[]),
        ):
            suggestions = analyze_reformulations(m, llm=True)
            assert isinstance(suggestions, list)


# ─────────────────────────────────────────────────────────────
# commentary.py
# ─────────────────────────────────────────────────────────────


class TestSolveCommentator:
    def test_creation(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Test model summary")
        assert c._model_summary == "Test model summary"
        assert c._interval == 10.0

    def test_custom_interval(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Model", interval=5.0)
        assert c._interval == 5.0

    def test_disabled_returns_none(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Model")
        c._enabled = False
        msg = c.maybe_comment(1.0, None, 0.0, None, 0, 0, 0)
        assert msg is None

    def test_get_root_comment_disabled(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Model")
        c._enabled = False
        msg = c.get_root_comment(objective=10.0, lower_bound=5.0, gap=0.5)
        assert msg is None

    def test_get_new_incumbent_comment_disabled(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Model")
        c._enabled = False
        msg = c.get_new_incumbent_comment(
            old_objective=20.0, new_objective=15.0, gap=0.1, node_count=50
        )
        assert msg is None

    def test_pending_message_returned(self):
        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Model")
        c._enabled = True
        c._pending_message = "Progress update!"
        msg = c.maybe_comment(1.0, None, 0.0, None, 0, 0, 0)
        assert msg == "Progress update!"
        assert c._pending_message is None

    def test_rate_limiting(self):
        import time

        from discopt.llm.commentary import SolveCommentator

        c = SolveCommentator("Model", interval=1000.0)
        c._enabled = True
        c._last_comment_time = time.monotonic()
        msg = c.maybe_comment(1.0, 10.0, 5.0, 0.5, 100, 50, 10)
        assert msg is None


# ─────────────────────────────────────────────────────────────
# chat.py
# ─────────────────────────────────────────────────────────────


class TestChatSession:
    def test_requires_litellm(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=False):
            with pytest.raises(ImportError, match="litellm"):
                ChatSession(verbose=False)

    def test_creation_with_mock_litellm(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=True):
            session = ChatSession(verbose=False)
            assert session.model is None
            assert session._active is True
            session.close()
            assert session._active is False

    def test_context_manager(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=True):
            with ChatSession(verbose=False) as session:
                assert session._active is True
            assert session._active is False

    def test_send_when_closed(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=True):
            session = ChatSession(verbose=False)
            session.close()
            response = session.send("hello")
            assert "closed" in response.lower()

    def test_send_with_mock_llm(self):
        from discopt.llm.chat import ChatSession

        mock_response = MagicMock()
        mock_response.tool_calls = None
        mock_response.content = "I can help you build a model."

        with patch("discopt.llm.is_available", return_value=True):
            session = ChatSession(verbose=False)
            with patch("discopt.llm.provider.complete_with_tools", return_value=mock_response):
                response = session.send("Help me build a model")
                assert response == "I can help you build a model."

    def test_solve_without_model(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=True):
            session = ChatSession(verbose=False)
            with pytest.raises(ValueError, match="No model"):
                session.solve()

    def test_extra_tools_present(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=True):
            session = ChatSession(verbose=False)
            tool_names = {t["function"]["name"] for t in session._tools}
            assert "solve_model" in tool_names
            assert "show_summary" in tool_names
            assert "validate_model" in tool_names
            assert "explain_result" in tool_names

    def test_system_prompt_content(self):
        from discopt.llm.chat import ChatSession

        with patch("discopt.llm.is_available", return_value=True):
            session = ChatSession(verbose=False)
            assert session._messages[0]["role"] == "system"
            assert "optimization" in session._messages[0]["content"].lower()


class TestChatFunction:
    def test_chat_returns_session(self):
        from discopt.llm.chat import chat

        with patch("discopt.llm.is_available", return_value=True):
            session = chat(verbose=False)
            assert isinstance(session, object)
            assert hasattr(session, "send")
            assert hasattr(session, "close")
            session.close()


# ─────────────────────────────────────────────────────────────
# provider.py
# ─────────────────────────────────────────────────────────────


class TestProvider:
    def test_get_model_default(self):
        from discopt.llm.provider import DEFAULT_MODEL, _get_model

        result = _get_model()
        assert result == DEFAULT_MODEL

    def test_get_model_explicit(self):
        from discopt.llm.provider import _get_model

        assert _get_model("openai/gpt-4o") == "openai/gpt-4o"

    def test_get_model_env_var(self):
        from discopt.llm.provider import _get_model

        with patch.dict("os.environ", {"DISCOPT_LLM_MODEL": "ollama/llama3"}):
            assert _get_model() == "ollama/llama3"

    def test_explicit_overrides_env(self):
        from discopt.llm.provider import _get_model

        with patch.dict("os.environ", {"DISCOPT_LLM_MODEL": "ollama/llama3"}):
            assert _get_model("openai/gpt-4o") == "openai/gpt-4o"

    def test_complete_requires_litellm(self):
        from discopt.llm.provider import complete

        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="litellm"):
                complete(messages=[{"role": "user", "content": "hi"}])

    def test_complete_with_mock_litellm(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_litellm.completion.return_value = mock_response

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            from importlib import reload

            import discopt.llm.provider as prov

            reload(prov)
            result = prov.complete(messages=[{"role": "user", "content": "hi"}])
            assert result == "Hello!"

    def test_complete_with_tools_requires_litellm(self):
        from discopt.llm.provider import complete_with_tools

        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="litellm"):
                complete_with_tools(
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[],
                )


# ─────────────────────────────────────────────────────────────
# __init__.py
# ─────────────────────────────────────────────────────────────


class TestLLMInit:
    def test_is_available_returns_bool(self):
        from discopt.llm import is_available

        assert isinstance(is_available(), bool)

    def test_get_completion_requires_litellm(self):
        from discopt.llm import get_completion

        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises((ImportError, RuntimeError)):
                get_completion(messages=[{"role": "user", "content": "hi"}])
