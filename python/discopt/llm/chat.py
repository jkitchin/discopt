"""
Conversational model building — ``discopt.chat()``.

Feature 3B: Stateful conversation loop where the user describes an
optimization problem in natural language and the LLM builds, modifies,
solves, and explains a discopt Model iteratively.

The Model class is fully mutable between solves, making iterative
refinement natural.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ChatSession:
    """Interactive conversational model building session.

    Maintains a conversation history and a mutable Model that the LLM
    modifies through tool calls.

    Parameters
    ----------
    llm_model : str, optional
        LLM model string (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
    verbose : bool, default True
        Print LLM responses to stdout.

    Examples
    --------
    >>> session = discopt.chat()
    >>> # Type: "I have a factory with 3 machines..."
    >>> # LLM asks clarifying questions, builds model, solves, explains
    >>> session.close()
    """

    def __init__(
        self,
        llm_model: str | None = None,
        verbose: bool = True,
    ):
        from discopt.llm import is_available

        if not is_available():
            raise ImportError(
                "litellm is required for discopt.chat(). Install it with: pip install discopt[llm]"
            )

        from discopt.llm.tools import TOOL_DEFINITIONS, ModelBuilder

        self._llm_model = llm_model
        self._verbose = verbose
        self._builder = ModelBuilder()
        self._tools = TOOL_DEFINITIONS + self._extra_tools()
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        self._active = True

    @property
    def model(self):
        """The current Model being built, or None."""
        return self._builder.model

    def send(self, message: str) -> str:
        """Send a message to the LLM and get a response.

        The LLM may call tools to modify the model, solve it, or
        ask clarifying questions. All tool calls are executed
        automatically.

        Parameters
        ----------
        message : str
            User message in natural language.

        Returns
        -------
        str
            The LLM's text response.
        """
        if not self._active:
            return "Session is closed. Create a new session with discopt.chat()."

        self._messages.append({"role": "user", "content": message})

        # Multi-turn tool-calling loop
        max_turns = 15
        for _ in range(max_turns):
            from discopt.llm.provider import complete_with_tools

            response = complete_with_tools(
                messages=self._messages,
                tools=self._tools,
                model=self._llm_model,
                max_tokens=4096,
                timeout=30.0,
            )

            tool_calls = getattr(response, "tool_calls", None)

            if not tool_calls:
                # LLM is done — extract text response
                content = getattr(response, "content", "") or ""
                self._messages.append({"role": "assistant", "content": content})
                if self._verbose and content:
                    print(content)
                return content

            # Execute tool calls
            self._messages.append(response)
            tool_results = self._execute_extended_tools(tool_calls)
            self._messages.extend(tool_results)

        return "Reached maximum conversation turns. Please try again."

    def solve(self, **kwargs) -> Any:
        r"""Solve the current model directly.

        Parameters
        ----------
        \*\*kwargs
            Arguments passed to ``Model.solve()``.

        Returns
        -------
        SolveResult
        """
        if self.model is None:
            raise ValueError("No model has been built yet.")
        return self.model.solve(**kwargs)

    def close(self):
        """Close the chat session."""
        self._active = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Internal ──

    def _system_prompt(self) -> str:
        """Build the system prompt for chat mode."""
        return (
            "You are an expert optimization consultant helping the user "
            "build and solve optimization models using discopt, a hybrid "
            "MINLP solver.\n\n"
            "You have tools to build models step by step. You also have "
            "tools to solve the model and show results.\n\n"
            "Guidelines:\n"
            "- Ask clarifying questions when the problem is ambiguous\n"
            "- Build the model incrementally, explaining each step\n"
            "- After building, validate and show the model summary\n"
            "- When asked to solve, use the solve_model tool\n"
            "- Explain results in plain language\n"
            "- The user can ask to modify the model between solves\n"
            "- Use descriptive variable and constraint names\n"
            "- Prefer indicator constraints over big-M when possible\n\n"
            "The discopt modeling API supports: continuous, binary, "
            "integer variables; minimize/maximize objectives; "
            "linear and nonlinear constraints; indicator constraints "
            "(if_then); disjunctive constraints (either_or); "
            "SOS1/SOS2 constraints; parametric optimization."
        )

    def _extra_tools(self) -> list[dict]:
        """Additional tools for chat mode beyond model building."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "solve_model",
                    "description": (
                        "Solve the current model. Returns the solve "
                        "result including status, objective, and solution."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_limit": {
                                "type": "number",
                                "description": "Time limit in seconds.",
                            },
                            "gap_tolerance": {
                                "type": "number",
                                "description": "Relative gap tolerance.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "show_summary",
                    "description": "Show the current model summary.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_model",
                    "description": (
                        "Validate the model for consistency "
                        "(check objective, variable names, bounds)."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explain_result",
                    "description": ("Explain the most recent solve result in plain language."),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def _execute_extended_tools(self, tool_calls) -> list[dict]:
        """Execute tool calls including chat-specific tools."""
        from discopt.llm.tools import execute_tool_calls

        results = []
        model_building_calls = []

        for call in tool_calls:
            fn_name = call.function.name

            if fn_name == "solve_model":
                results.append(self._handle_solve(call))
            elif fn_name == "show_summary":
                results.append(self._handle_summary(call))
            elif fn_name == "validate_model":
                results.append(self._handle_validate(call))
            elif fn_name == "explain_result":
                results.append(self._handle_explain(call))
            else:
                model_building_calls.append(call)

        # Execute model building tools
        if model_building_calls:
            results.extend(execute_tool_calls(model_building_calls, self._builder))

        return results

    def _handle_solve(self, call) -> dict:
        """Handle solve_model tool call."""
        if self.model is None:
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "Error: No model built yet. Create a model first.",
            }

        try:
            args = json.loads(call.function.arguments) if call.function.arguments else {}
        except json.JSONDecodeError:
            args = {}

        try:
            self.model.validate()
            result = self.model.solve(
                time_limit=args.get("time_limit", 300),
                gap_tolerance=args.get("gap_tolerance", 1e-4),
            )
            self._last_result = result

            from discopt.llm.serializer import serialize_solve_result

            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": serialize_solve_result(result, self.model),
            }
        except Exception as e:
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"Solve failed: {e}",
            }

    def _handle_summary(self, call) -> dict:
        """Handle show_summary tool call."""
        if self.model is None:
            content = "No model built yet."
        else:
            content = self.model.summary()
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def _handle_validate(self, call) -> dict:
        """Handle validate_model tool call."""
        if self.model is None:
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "No model built yet.",
            }
        try:
            self.model.validate()
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": "Model is valid.",
            }
        except ValueError as e:
            return {
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"Validation error: {e}",
            }

    def _handle_explain(self, call) -> dict:
        """Handle explain_result tool call."""
        result = getattr(self, "_last_result", None)
        if result is None:
            content = "No solve result available. Solve the model first."
        else:
            content = result.explain()
        return {"role": "tool", "tool_call_id": call.id, "content": content}


def chat(
    llm_model: str | None = None,
    verbose: bool = True,
) -> ChatSession:
    """Start an interactive model building session.

    Parameters
    ----------
    llm_model : str, optional
        LLM model string.
    verbose : bool, default True
        Print LLM responses to stdout.

    Returns
    -------
    ChatSession

    Examples
    --------
    >>> import discopt
    >>> session = discopt.chat()
    >>> session.send("I need to minimize shipping costs from 3 warehouses")
    >>> session.send("Each warehouse has capacity [100, 150, 200]")
    >>> session.send("Solve it")
    >>> session.close()
    """
    return ChatSession(llm_model=llm_model, verbose=verbose)
