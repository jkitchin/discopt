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
    NLP solver backends: POUNCE (pure-Rust Ipopt port), pure-JAX IPM (vmap batch), cyipopt (Ipopt).
"""

__version__ = "0.4.1.dev0"

# Enable JAX 64-bit mode before any downstream discopt import triggers a
# jax import. IPOPT tolerances (default tol=1e-6, bound_relax_factor=1e-8)
# are incompatible with float32 residuals — silent truncation to float32
# causes NMPC failures and parameter-array warnings. Users who need float32
# can opt out by setting ``JAX_ENABLE_X64=0`` in the environment before
# importing discopt.
#
# We set the environment variable rather than importing jax here: JAX reads
# ``JAX_ENABLE_X64`` at its own (lazy) import time, so this keeps ``import
# discopt`` free of the multi-second jax import for code paths (e.g. the GAMS
# link's LP/MILP solves) that never touch jax.
import os as _os

if _os.environ.get("JAX_ENABLE_X64", "1") != "0":
    _os.environ.setdefault("JAX_ENABLE_X64", "1")

# Enable JAX's persistent (on-disk) compilation cache. A solve recompiles a
# handful of small XLA kernels (relaxation evaluators, NLP residual/Jacobian,
# McCormick envelopes); each is only ~15 ms but they are re-traced from scratch
# in every fresh process, so a short solve pays a fixed compile tax on every
# run. Caching the compiled artifacts keyed by HLO + backend + XLA version lets
# subsequent processes load them instead of recompiling — measured ~45% wall
# reduction on cold fresh-process solves (e.g. nvs06 1.5 s -> 0.82 s). The cache
# is purely a build-artifact reuse: it keys on the exact computation, so it can
# never change a solve's numerical result — correctness-neutral.
#
# As with JAX_ENABLE_X64 we only set environment variables (no jax import here)
# to keep ``import discopt`` free of the multi-second jax import. Notes:
#   * Respect a user-provided JAX_COMPILATION_CACHE_DIR (don't override it).
#   * Allow opt-out via DISCOPT_DISABLE_JAX_CACHE=1.
#   * JAX's default MIN_COMPILE_TIME_SECS is 1.0 s, which would skip discopt's
#     ~15 ms compiles entirely, and MIN_ENTRY_SIZE_BYTES defaults nonzero — set
#     both to 0 so the small-but-frequent kernels actually get cached.
if (
    _os.environ.get("JAX_COMPILATION_CACHE_DIR") is None
    and _os.environ.get("DISCOPT_DISABLE_JAX_CACHE", "0") == "0"
):
    _cache_root = _os.environ.get("XDG_CACHE_HOME") or _os.path.join(
        _os.path.expanduser("~"), ".cache"
    )
    _jax_cache = _os.path.join(_cache_root, "discopt", "jax-cache")
    try:
        _os.makedirs(_jax_cache, exist_ok=True)
    except OSError:
        # Unwritable cache location (read-only home, sandbox): skip silently —
        # the cache is a pure optimization, absence only costs recompilation.
        pass
    else:
        _os.environ["JAX_COMPILATION_CACHE_DIR"] = _jax_cache
        _os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
        _os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

from discopt.callbacks import (
    CallbackContext as CallbackContext,
)
from discopt.callbacks import (
    CutResult as CutResult,
)
from discopt.decomposition import (
    DecompositionStructure as DecompositionStructure,
)
from discopt.decomposition import (
    detect_decomposition as detect_decomposition,
)
from discopt.infeasibility import (
    IISResult as IISResult,
)
from discopt.infeasibility import (
    compute_iis as compute_iis,
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
    ProductSet as ProductSet,
)
from discopt.modeling import (
    RangeSet as RangeSet,
)
from discopt.modeling import (
    Set as Set,
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
    example_assignment as example_assignment,
)
from discopt.modeling.examples import (
    example_multicommodity_flow as example_multicommodity_flow,
)
from discopt.modeling.examples import (
    example_simple_minlp as example_simple_minlp,
)
from discopt.modeling.examples import (
    example_transportation as example_transportation,
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
