"""
discopt -- Mixed-Integer Nonlinear Programming with JAX and Rust.

A hybrid MINLP solver combining a Rust backend (LP solving, Branch & Bound tree
management), JAX (automatic differentiation, NLP relaxations, GPU acceleration),
and Python orchestration.

Quick Start
-----------
>>> import discopt
>>> m = discopt.Model("example")
>>> x = m.continuous("x", shape=(2,), lb=0, ub=10)
>>> y = m.binary("y")
>>> m.minimize(x[0] + 2 * x[1] + 5 * y)
>>> m.subject_to(x[0] + x[1] >= 3)
>>> result = m.solve()

Submodules
----------
modeling
    Model building API: Model, Variable, Expression, Constraint, math functions.
solver
    Solve orchestrator: Branch & Bound with NLP relaxations.
solvers
    NLP solver backends: ripopt (Rust IPM), pure-JAX IPM (vmap batch), cyipopt (Ipopt).
"""

__version__ = "0.2.5"

from discopt.callbacks import (
    CallbackContext as CallbackContext,
)
from discopt.callbacks import (
    CutResult as CutResult,
)
from discopt.modeling import (
    Constraint as Constraint,
)
from discopt.modeling import (
    Expression as Expression,
)
from discopt.modeling import (
    Model as Model,
)
from discopt.modeling import (
    Parameter as Parameter,
)
from discopt.modeling import (
    SolveResult as SolveResult,
)
from discopt.modeling import (
    Variable as Variable,
)
from discopt.modeling import (
    VarType as VarType,
)
from discopt.modeling import (
    cos as cos,
)
from discopt.modeling import (
    exp as exp,
)
from discopt.modeling import (
    log as log,
)
from discopt.modeling import (
    sin as sin,
)
from discopt.modeling import (
    sqrt as sqrt,
)
from discopt.modeling import (
    tan as tan,
)
from discopt.modeling.examples import (
    example_simple_minlp as example_simple_minlp,
)

# Lazy imports for optional modules (avoid import overhead at startup)


def estimate_parameters(*args, **kwargs):
    """Estimate unknown parameters from experimental data.

    See :func:`discopt.estimate.estimate_parameters` for full documentation.
    """
    from discopt.estimate import estimate_parameters as _ep

    return _ep(*args, **kwargs)


def chat(llm_model: str | None = None, verbose: bool = True):
    """Start an interactive LLM-powered model building session.

    Requires ``pip install discopt[llm]``.

    Parameters
    ----------
    llm_model : str, optional
        LLM model string (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
    verbose : bool, default True
        Print LLM responses to stdout.

    Returns
    -------
    ChatSession
    """
    from discopt.llm.chat import chat as _chat

    return _chat(llm_model=llm_model, verbose=verbose)
