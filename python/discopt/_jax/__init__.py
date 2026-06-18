"""JAX compilation utilities for discopt.

The DAG compiler (and therefore ``jax``) is imported **lazily** via PEP 562
``__getattr__``: importing a JAX-free submodule such as ``discopt._jax.deadline``
no longer drags in ``dag_compiler`` and the heavy JAX/XLA initialization. The
public ``compile_*`` helpers resolve on first attribute access, so
``discopt._jax.compile_expression`` still works unchanged.
"""

from typing import TYPE_CHECKING

__all__ = ["compile_expression", "compile_objective", "compile_constraint"]

if TYPE_CHECKING:
    from discopt._jax.dag_compiler import (
        compile_constraint,
        compile_expression,
        compile_objective,
    )


def __getattr__(name: str) -> object:
    if name in __all__:
        from discopt._jax import dag_compiler

        return getattr(dag_compiler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
