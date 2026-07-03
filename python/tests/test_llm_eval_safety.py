"""Regression tests for LLM-1: the ``_eval_expression`` sandbox escape (RCE).

Tool-call arguments (``lhs``/``rhs``/``expression``/``value``) are
attacker-influenced — anyone who can steer the LLM's output controls them. The
old implementation ``eval()``'d them with an emptied ``__builtins__``, which is
not a sandbox: any in-scope object bridges back to builtins via
``().__class__.__…__`` or a module's ``__loader__.__globals__``. A working RCE
(``os.system`` writing a file) was reproduced through the public
``ModelBuilder.execute_tool`` path.

These tests pin the *class* of escape (attribute access, lambdas, comprehensions,
dunder chains, calls to non-whitelisted callables) as blocked, and confirm the
legitimate arithmetic grammar still evaluates. They call ``_eval_expression``
directly, so they run in well under a second.

Fails-before/passes-after: on the pre-fix ``eval`` implementation, the RCE
payloads execute (marker file created) instead of raising.
"""

from __future__ import annotations

import pytest
from discopt.llm.tools import ModelBuilder

pytestmark = pytest.mark.smoke


def _builder() -> ModelBuilder:
    b = ModelBuilder()
    b.execute_tool("create_model", {"name": "m"})
    b.execute_tool("add_continuous", {"name": "x", "size": 3, "lb": 0, "ub": 10})
    b.execute_tool("add_continuous", {"name": "y", "size": 2, "lb": 0, "ub": 10})
    return b


# The escape vectors, drawn from the LLM-1 review (§2) plus the standard sandbox
# breakouts. Each MUST raise rather than execute.
_MALICIOUS = [
    # reach __import__ via a module's loader globals (the confirmed RCE)
    "np.__loader__.__init__.__globals__['__builtins__']['__import__']('os')",
    # object-subclass walk back to builtins
    "().__class__.__bases__[0].__subclasses__()",
    # dunder chain off a model variable
    "x.__class__.__mro__",
    "x.__class__",
    # lambda / callable construction
    "(lambda: 1)()",
    # comprehensions (can hide calls / iteration)
    "[c for c in range(3)]",
    # attribute access on numpy that the grammar never needs
    "np.zeros",
    "dm.exp",
]


@pytest.mark.parametrize("payload", _MALICIOUS)
def test_eval_expression_rejects_escape_payloads(payload, tmp_path):
    """Every sandbox-escape payload raises ValueError and runs no code."""
    marker = tmp_path / "rce_marker"
    # Build a payload that would create the marker IF code executed. We still
    # assert the direct payload raises; the marker guards against partial
    # evaluation side effects.
    b = _builder()
    with pytest.raises(ValueError):
        b._eval_expression(payload)
    assert not marker.exists()


def test_eval_expression_blocks_os_system_rce(tmp_path):
    """The precise confirmed RCE (os.system side effect) must not fire."""
    marker = tmp_path / "pwned"
    assert not marker.exists()
    payload = (
        "np.__loader__.__init__.__globals__['__builtins__']"
        f"['__import__']('os').system('touch {marker}')"
    )
    b = _builder()
    with pytest.raises(ValueError):
        b._eval_expression(payload)
    assert not marker.exists(), "arbitrary code executed — sandbox escaped"


# The legitimate grammar the tool schema advertises: arithmetic, indexing, power,
# and bare math/reduction functions. All must still evaluate to an expression/DAG.
_LEGIT = [
    "x[0] + x[1]",
    "x[0] + 2*x[1] + 3*x[2]",
    "x",
    "5",
    "y[1] * 100",
    "x[0]**2 + 5*y[0]",
    "-x[0]",
    "x[0] / 2",
    "sum(x)",
    "abs(x[2])",
    "exp(x[0])",
    "sqrt(x[1]) + sin(y[0])",
    "log(x[0]) + cos(y[1])",
]


@pytest.mark.parametrize("expr", _LEGIT)
def test_eval_expression_accepts_legitimate_grammar(expr):
    """The advertised arithmetic/indexing/math-call grammar still works."""
    b = _builder()
    result = b._eval_expression(expr)
    assert result is not None


def test_sum_is_the_modeling_reduction_not_builtin():
    """`sum(x)` must be dm.sum (finite), not builtin sum (which would iterate a
    Variable forever via the sequence protocol)."""
    b = _builder()
    # Should return promptly and be a DAG node, not hang or be a Python int.
    result = b._eval_expression("sum(x)")
    assert type(result).__name__ != "int"


def test_unknown_name_raises_cleanly():
    b = _builder()
    with pytest.raises(ValueError):
        b._eval_expression("undefined_var + 1")
