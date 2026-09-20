"""Deprecated alias for :mod:`discopt.ml`.

The package was renamed in 0.9.0: it embeds *machine-learning* predictors —
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
import importlib.abc
import importlib.machinery

# #1325: ``find_spec`` below is ``importlib.util.find_spec``. It worked only
# because something else in the process had already imported ``importlib.util``
# -- ``import importlib`` does not bring the submodule in -- so the guard's
# ``except (ImportError, AttributeError, ValueError)`` would have turned "the
# submodule is missing" into "the alias does not exist", silently.
import importlib.util
import logging
import sys
import warnings

import discopt.ml as _ml

_MESSAGE = (
    "discopt.nn is deprecated and will be removed in 0.10; use discopt.ml "
    "instead (the package embeds ML predictors generally — trees and other "
    "surrogates as well as neural networks). See issue #1219."
)

warnings.warn(_MESSAGE, DeprecationWarning, stacklevel=2)

# #1314: ``DeprecationWarning`` is ignored by Python's default filters outside
# ``__main__``, so the ``warnings.warn`` above is invisible for the case that
# actually matters — a downstream *library* module doing ``import discopt.nn``.
# The shim's guarantee is "it still imports, and warns"; a warning only a test
# harness (which overrides those filters) can see does not meet it. Logging is
# the second channel, and it reaches a user who has configured no logging at all
# via ``logging.lastResort``, which prints WARNING and above to stderr — the same
# reasoning as the no-dual-bound notice in ``Model.solve``. Both channels carry
# the identical text, and neither mutates the process-global warning filters: a
# library that installs its own filter changes what *every* other library's
# warnings do.
logging.getLogger(__name__).warning("%s", _MESSAGE)


class _AliasLoader(importlib.abc.Loader):
    """Loader that hands back an already-imported ``discopt.ml`` module.

    ``create_module`` returning the real module object (rather than a fresh one)
    is what makes the alias a rename and not a fork: ``sys.modules`` ends up with
    both names bound to one object, so ``isinstance`` and class identity hold
    across the two spellings. ``exec_module`` is a no-op precisely because that
    object has already been executed — re-executing it would create a second set
    of classes under the same names, which is the failure this shim exists to
    prevent.
    """

    def __init__(self, real_name: str) -> None:
        self._real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self._real_name)

    def exec_module(self, module) -> None:
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``discopt.nn.<sub>`` to ``discopt.ml.<sub>``, on demand.

    #1314: this used to be an eager ``pkgutil.walk_packages`` that imported the
    *entire* ``discopt.ml`` tree at ``import discopt.nn`` time — including
    ``readers/*``, which ``import discopt.ml`` itself never touches (they are
    reachable only through the lazy ``load_*`` wrappers). That made the
    deprecated path strictly less robust than the one it forwards to: a single
    eager third-party import anywhere under ``discopt.ml`` would break ``import
    discopt.nn`` for every caller while leaving ``import discopt.ml`` working.
    A deprecated compatibility shim must be at least as robust as its target,
    never less.

    Resolving lazily also restores the pre-rename behaviour: the old ``nn``
    package's ``__init__`` was exactly as lazy as ``discopt.ml.__init__`` is now.
    Each alias is imported only when something actually asks for it, and a
    failure to import surfaces at that point, attributed to that module, exactly
    as it would under the ``discopt.ml`` spelling.
    """

    _PREFIX = f"{__name__}."
    #: See ``_ALIAS_FINDER_MARK``: identity across a reload of this module.
    _discopt_nn_alias_finder = True

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith(self._PREFIX):
            return None
        real_name = f"{_ml.__name__}.{fullname[len(self._PREFIX) :]}"
        # Only claim names that really exist under discopt.ml. Returning a spec
        # for a bogus one would turn a ModuleNotFoundError naming the module into
        # one raised from inside this loader.
        try:
            if importlib.util.find_spec(real_name) is None:
                return None
        except (ImportError, AttributeError, ValueError):
            return None
        spec = importlib.machinery.ModuleSpec(fullname, _AliasLoader(real_name))
        # Submodules of an aliased package must keep resolving through this
        # finder, so the alias has to look like a package whenever its target is.
        real_path = getattr(sys.modules.get(real_name), "__path__", None)
        if real_path is None:
            parent_spec = importlib.util.find_spec(real_name)
            real_path = getattr(parent_spec, "submodule_search_locations", None)
        if real_path is not None:
            spec.submodule_search_locations = list(real_path)
        return spec


#: Attribute marking an installed alias finder, checked instead of
#: ``isinstance``. #1325: re-executing this module (``importlib.reload``, or a
#: ``sys.modules`` purge and re-import) creates a NEW ``_AliasFinder`` class, so
#: the instance already on ``sys.meta_path`` is not an instance of it and the
#: guard installed a second finder -- then a third. A name, unlike a class
#: object, survives the module being rebuilt.
_ALIAS_FINDER_MARK = "_discopt_nn_alias_finder"


def _install_alias_finder() -> None:
    """Put the alias finder ahead of the normal path finders, exactly once.

    First in ``sys.meta_path`` because the standard finders would otherwise look
    for ``discopt/nn/network.py``, a file that no longer exists.
    """
    if not any(getattr(f, _ALIAS_FINDER_MARK, False) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())


_install_alias_finder()

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
