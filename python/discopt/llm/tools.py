"""
Tool definitions for LLM structured output via function calling.

These tools map directly to Model methods — the LLM calls tools,
and each tool invokes the corresponding Model method. No arbitrary
code execution.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from discopt.modeling.core import Model

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Tool definitions (OpenAI function-calling format)
# ─────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "create_model",
            "description": "Create a new optimization model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Descriptive name for the model.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_continuous",
            "description": (
                "Add continuous decision variable(s) to the model. "
                "Use shape for vector/matrix variables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name (must be unique).",
                    },
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Shape as array of ints. "
                            "Empty array [] for scalar, [n] for vector, [m,n] for matrix."
                        ),
                    },
                    "lb": {
                        "type": "number",
                        "description": "Lower bound (default: -1e20 = unbounded).",
                    },
                    "ub": {
                        "type": "number",
                        "description": "Upper bound (default: 1e20 = unbounded).",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_binary",
            "description": "Add binary (0/1) decision variable(s) to the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name (must be unique).",
                    },
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Shape as array of ints. Empty [] for scalar, [n] for vector."
                        ),
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_integer",
            "description": "Add integer decision variable(s) to the model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Variable name (must be unique).",
                    },
                    "shape": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Shape as array of ints.",
                    },
                    "lb": {
                        "type": "integer",
                        "description": "Lower bound.",
                    },
                    "ub": {
                        "type": "integer",
                        "description": "Upper bound.",
                    },
                },
                "required": ["name", "lb", "ub"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_parameter",
            "description": (
                "Add a named parameter (fixed value) to the model. "
                "Parameters are constants that can be used in expressions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Parameter name.",
                    },
                    "value": {
                        "description": "Parameter value (number or array of numbers).",
                    },
                },
                "required": ["name", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_objective",
            "description": (
                "Set the objective function. Expression is a string describing "
                "the objective in terms of variable names and math operations. "
                "Supported: +, -, *, /, **, sum(), exp(), log(), sqrt(), sin(), cos()."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sense": {
                        "type": "string",
                        "enum": ["minimize", "maximize"],
                        "description": "Optimization direction.",
                    },
                    "expression": {
                        "type": "string",
                        "description": (
                            "Objective expression using variable names. "
                            "E.g., 'x[0]**2 + x[1]**2 + 5*y'"
                        ),
                    },
                },
                "required": ["sense", "expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_constraint",
            "description": (
                "Add a constraint to the model. "
                "Expression is the left-hand side, sense is <=, ==, or >=, "
                "rhs is the right-hand side value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lhs": {
                        "type": "string",
                        "description": (
                            "Left-hand side expression using variable names. E.g., 'x[0] + x[1]'"
                        ),
                    },
                    "sense": {
                        "type": "string",
                        "enum": ["<=", "==", ">="],
                        "description": "Constraint sense.",
                    },
                    "rhs": {
                        "type": "string",
                        "description": ("Right-hand side expression. E.g., '10' or 'y * 100'"),
                    },
                    "name": {
                        "type": "string",
                        "description": "Descriptive constraint name.",
                    },
                },
                "required": ["lhs", "sense", "rhs", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_if_then",
            "description": (
                "Add an indicator constraint: when binary indicator = 1, "
                "the inner constraints must hold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {
                        "type": "string",
                        "description": "Name of the binary indicator variable.",
                    },
                    "constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lhs": {"type": "string"},
                                "sense": {
                                    "type": "string",
                                    "enum": ["<=", "==", ">="],
                                },
                                "rhs": {"type": "string"},
                            },
                            "required": ["lhs", "sense", "rhs"],
                        },
                        "description": "List of constraints active when indicator = 1.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Descriptive name for the indicator constraint.",
                    },
                },
                "required": ["indicator", "constraints", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_either_or",
            "description": (
                "Add a disjunctive constraint: exactly one group of constraints "
                "must hold. Each group is a list of constraints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "disjuncts": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "lhs": {"type": "string"},
                                    "sense": {
                                        "type": "string",
                                        "enum": ["<=", "==", ">="],
                                    },
                                    "rhs": {"type": "string"},
                                },
                                "required": ["lhs", "sense", "rhs"],
                            },
                        },
                        "description": (
                            "List of disjunct groups. Each group is a list of "
                            "constraints that must hold together."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": "Descriptive name for the disjunction.",
                    },
                },
                "required": ["disjuncts", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_implies",
            "description": (
                "Add a logical implication: y1 = 1 implies y2 = 1. Both must be binary variables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y1": {
                        "type": "string",
                        "description": "Name of the antecedent binary variable.",
                    },
                    "y2": {
                        "type": "string",
                        "description": "Name of the consequent binary variable.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Descriptive name for the implication.",
                    },
                },
                "required": ["y1", "y2", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_at_least",
            "description": (
                "Add a cardinality constraint: at least k of the given binary variables must be 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "k": {
                        "type": "integer",
                        "description": "Minimum number of binaries that must be 1.",
                    },
                    "binaries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of binary variables.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Descriptive name for the constraint.",
                    },
                },
                "required": ["k", "binaries", "name"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────
# Tool execution engine
# ─────────────────────────────────────────────────────────────


class ModelBuilder:
    """Executes LLM tool calls to build a Model incrementally.

    Each tool call maps to a Model method. The builder maintains the model
    and a namespace of variables/parameters for expression evaluation.
    """

    def __init__(self):
        self.model: Model | None = None
        self._namespace: dict[str, Any] = {}

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a single tool call and return a status message.

        Parameters
        ----------
        tool_name : str
            Name of the tool to execute.
        args : dict
            Tool arguments (already parsed from JSON).

        Returns
        -------
        str
            Status message for the LLM.
        """
        from discopt.llm.safety import sanitize_tool_args

        args = sanitize_tool_args(tool_name, args)

        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            return str(handler(args))
        except Exception as e:
            logger.warning("Tool '%s' failed: %s", tool_name, e)
            return f"Error executing {tool_name}: {e}"

    def _handle_create_model(self, args: dict) -> str:
        self.model = Model(name=args["name"])
        # Pre-populate namespace with dm math functions
        import discopt.modeling as dm

        self._namespace["dm"] = dm
        self._namespace["np"] = np
        return f"Created model '{args['name']}'"

    def _handle_add_continuous(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        shape = tuple(args.get("shape", []))
        lb = args.get("lb", -1e20)
        ub = args.get("ub", 1e20)
        var = self.model.continuous(args["name"], shape=shape, lb=lb, ub=ub)
        self._namespace[args["name"]] = var
        return f"Added continuous variable '{args['name']}' shape={shape}"

    def _handle_add_binary(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        shape = tuple(args.get("shape", []))
        var = self.model.binary(args["name"], shape=shape)
        self._namespace[args["name"]] = var
        return f"Added binary variable '{args['name']}' shape={shape}"

    def _handle_add_integer(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        shape = tuple(args.get("shape", []))
        var = self.model.integer(args["name"], shape=shape, lb=args["lb"], ub=args["ub"])
        self._namespace[args["name"]] = var
        return f"Added integer variable '{args['name']}' lb={args['lb']} ub={args['ub']}"

    def _handle_add_parameter(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        value = args["value"]
        if isinstance(value, list):
            value = np.array(value)
        param = self.model.parameter(args["name"], value=value)
        self._namespace[args["name"]] = param
        return f"Added parameter '{args['name']}'"

    def _handle_set_objective(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        expr = self._eval_expression(args["expression"])
        if args["sense"] == "minimize":
            self.model.minimize(expr)
        else:
            self.model.maximize(expr)
        return f"Set objective: {args['sense']} {args['expression']}"

    def _handle_add_constraint(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        lhs = self._eval_expression(args["lhs"])
        rhs = self._eval_expression(args["rhs"])
        sense = args["sense"]
        name = args.get("name")

        if sense == "<=":
            self.model.subject_to(lhs <= rhs, name=name)
        elif sense == ">=":
            self.model.subject_to(lhs >= rhs, name=name)
        elif sense == "==":
            self.model.subject_to(lhs == rhs, name=name)
        else:
            return f"Error: invalid sense '{sense}'"

        return f"Added constraint '{name}': {args['lhs']} {sense} {args['rhs']}"

    def _handle_add_if_then(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        indicator = self._namespace.get(args["indicator"])
        if indicator is None:
            return f"Error: indicator variable '{args['indicator']}' not found"

        constraints = []
        for c in args["constraints"]:
            lhs = self._eval_expression(c["lhs"])
            rhs = self._eval_expression(c["rhs"])
            sense = c["sense"]
            if sense == "<=":
                constraints.append(lhs <= rhs)
            elif sense == ">=":
                constraints.append(lhs >= rhs)
            elif sense == "==":
                constraints.append(lhs == rhs)

        self.model.if_then(indicator, constraints, name=args.get("name"))
        return f"Added indicator constraint '{args.get('name')}'"

    def _handle_add_either_or(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        disjuncts = []
        for group in args["disjuncts"]:
            constraints = []
            for c in group:
                lhs = self._eval_expression(c["lhs"])
                rhs = self._eval_expression(c["rhs"])
                sense = c["sense"]
                if sense == "<=":
                    constraints.append(lhs <= rhs)
                elif sense == ">=":
                    constraints.append(lhs >= rhs)
                elif sense == "==":
                    constraints.append(lhs == rhs)
            disjuncts.append(constraints)
        self.model.either_or(disjuncts, name=args.get("name"))
        return f"Added disjunctive constraint '{args.get('name')}'"

    def _handle_add_implies(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        y1 = self._namespace.get(args["y1"])
        if y1 is None:
            return f"Error: variable '{args['y1']}' not found"
        y2 = self._namespace.get(args["y2"])
        if y2 is None:
            return f"Error: variable '{args['y2']}' not found"
        self.model.implies(y1, y2, name=args.get("name"))
        return f"Added implication '{args.get('name')}': {args['y1']} => {args['y2']}"

    def _handle_add_at_least(self, args: dict) -> str:
        if self.model is None:
            return "Error: call create_model first"
        binaries = []
        for name in args["binaries"]:
            var = self._namespace.get(name)
            if var is None:
                return f"Error: variable '{name}' not found"
            binaries.append(var)
        self.model.at_least(args["k"], binaries, name=args.get("name"))
        return f"Added at_least({args['k']}) constraint '{args.get('name')}'"

    def _eval_expression(self, expr_str: str) -> Any:
        """Safely evaluate an expression string in the model namespace.

        Only allows access to model variables, parameters, numpy, and dm
        math functions. No builtins or arbitrary code execution.
        """
        # Build a restricted namespace
        safe_ns = {
            "__builtins__": {},
            "sum": sum,
            "range": range,
            "abs": abs,
            "min": min,
            "max": max,
        }
        safe_ns.update(self._namespace)

        try:
            return eval(expr_str, safe_ns)  # noqa: S307
        except Exception as e:
            raise ValueError(f"Cannot evaluate expression '{expr_str}': {e}") from e


def execute_tool_calls(tool_calls: list, builder: ModelBuilder) -> list[dict]:
    """Execute a batch of tool calls from the LLM response.

    Parameters
    ----------
    tool_calls : list
        Tool calls from the LLM response message.
    builder : ModelBuilder
        The model builder to execute tools on.

    Returns
    -------
    list of dict
        Tool results as messages for the conversation.
    """
    results = []
    for call in tool_calls:
        fn = call.function
        tool_name = fn.name
        try:
            args = json.loads(fn.arguments)
        except json.JSONDecodeError as e:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": f"Error parsing arguments: {e}",
                }
            )
            continue

        result = builder.execute_tool(tool_name, args)
        results.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
        )

    return results
