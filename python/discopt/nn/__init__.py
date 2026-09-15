"""Deprecated alias for :mod:`discopt.ml`.

The package was renamed in 0.8.1: it embeds *machine-learning* predictors —
decision trees and tree ensembles as much as neural networks, plus anything
satisfying the :class:`~discopt.ml.surrogate.Surrogate` protocol — so the old
``nn`` name described one of the families it supports (issue #1219).

Importing ``discopt.nn`` emits a :class:`DeprecationWarning` and forwards to
:mod:`discopt.ml`. The forwarding is by object identity, not by copy: every
submodule path (``discopt.nn.network``, ``discopt.nn.formulations.base``,
``discopt.nn.readers.sklearn_reader``, ...) resolves to the *same* module
object as its ``discopt.ml`` counterpart, so ``isinstance`` checks and
class identity hold across the two spellings. Update imports to
``discopt.ml``; this shim is scheduled for removal in 0.10.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

import discopt.ml as _ml

warnings.warn(
    "discopt.nn is deprecated and will be removed in 0.10; use discopt.ml "
    "instead (the package embeds ML predictors generally — trees and other "
    "surrogates as well as neural networks). See issue #1219.",
    DeprecationWarning,
    stacklevel=2,
)


def _install_submodule_aliases() -> None:
    """Register every ``discopt.ml`` submodule under its ``discopt.nn`` name.

    Eager rather than lazy: the alias entries must already be in
    ``sys.modules`` when the import machinery resolves ``import
    discopt.nn.network``, otherwise it looks for a file that no longer exists.
    Walking the real package instead of hardcoding a list means a module added
    to ``discopt.ml`` later is aliased without touching this file.

    No module under ``discopt.ml`` imports a third-party dependency at import
    time — the optional readers defer ``onnx`` / ``sklearn`` / ``torch`` into
    their functions — so walking the whole package is cheap and cannot fail on
    a missing extra. ``onerror`` re-raises rather than accepting
    ``walk_packages``' default of swallowing an ``ImportError``: a subpackage
    that failed to import would otherwise drop itself *and its children* from
    the alias set, and the only symptom would be a ``ModuleNotFoundError`` at
    some later ``import discopt.nn.<something>``.
    """

    def _reraise(name: str) -> None:
        # Called from inside walk_packages' own ``except ImportError``, so a
        # bare raise re-raises that error (verified: with onerror=None the
        # failing subpackage *and* its children vanish from the walk silently).
        raise

    for info in pkgutil.walk_packages(_ml.__path__, prefix=f"{_ml.__name__}.", onerror=_reraise):
        module = importlib.import_module(info.name)
        alias = f"{__name__}{info.name[len(_ml.__name__) :]}"
        sys.modules[alias] = module


_install_submodule_aliases()

# Re-export the full public surface. Forwarding by ``getattr`` on the real
# module keeps the two spellings from drifting: a name added to ``discopt.ml``
# is reachable here with no edit to this file, and every object is the same
# object, not a copy.
_FORWARDED = (
    *_ml.__all__,
    # Deferred loaders for the optional readers; public, but kept out of
    # ``discopt.ml.__all__`` because each imports a third-party dependency.
    "add_predictor",
    "load_onnx",
    "load_sklearn_ensemble",
    "load_sklearn_mlp",
    "load_sklearn_tree",
    "load_torch_sequential",
)

__all__ = list(_FORWARDED)


def __getattr__(name: str) -> object:
    """Forward any public ``discopt.ml`` attribute, including ones added later."""
    if not name.startswith("_"):
        try:
            return getattr(_ml, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*__all__, *dir(_ml)})
