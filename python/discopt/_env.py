"""One truth table for every ``DISCOPT_*`` environment flag.

Before this module the repo had **seven incompatible boolean parse idioms**
(architecture review 2026-07-28 §2.4), so the same string meant opposite things
depending on which flag you set it on: ``DISCOPT_RLT=false`` turned RLT *on*
(``raw != "0"``), ``DISCOPT_CONVEX_KERNEL=off`` turned the kernel *on*
(``not in ("0", "", "false", "False")``), and ``DISCOPT_SGO=2`` turned SGO *off*
(``in ("1", "true", "yes", "on")``). Empty string was true under three idioms and
false under four.

Every flag read now goes through this module, with one table:

======================================  =========================================
value (case-insensitive, stripped)      result
======================================  =========================================
``1`` / ``true`` / ``yes`` / ``on``     ``True``
``0`` / ``false`` / ``no`` / ``off``    ``False``
unset, or empty/whitespace-only         the caller's ``default``
anything else                           ``ValueError`` (loud refusal, CLAUDE.md §3)
======================================  =========================================

A typo (``DISCOPT_RLT=ture``) used to silently pick an arm; it now raises with the
flag name and the accepted values. Graduated flags keep their ``=0`` opt-out —
``0`` is in the false column, as it always was.

Call sites must pass a **string literal** flag name so the registry test
(``test_flag_registry.py``) can grep them; every literal name must have a row in
:data:`discopt._flag_registry.FLAG_REGISTRY`. The one exception is the daemon
config (``_daemon_core._resolve``), which builds ``DISCOPT_SOLVE_*`` /
``DISCOPT_GAMS_*`` names by f-string from a prefix; those names are registered
fully expanded and covered by their own test.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, overload

__all__ = [
    "TRUE_VALUES",
    "FALSE_VALUES",
    "env_bool",
    "env_int",
    "env_float",
    "env_str",
    "env_enum",
    "env_is_set",
]

#: Accepted true spellings (compared case-insensitively, after stripping).
TRUE_VALUES = ("1", "true", "yes", "on")
#: Accepted false spellings (compared case-insensitively, after stripping).
FALSE_VALUES = ("0", "false", "no", "off")

_TRUE_SET = frozenset(TRUE_VALUES)
_FALSE_SET = frozenset(FALSE_VALUES)


def _raw(name: str) -> Optional[str]:
    """The stripped value of ``name``, or ``None`` when unset or blank.

    Blank is folded into unset deliberately: ``FOO=`` in a shell profile is how
    people *unset* a flag, and the old idioms disagreed about it (three read it as
    true, four as false).
    """
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def env_is_set(name: str) -> bool:
    """Whether ``name`` is present in the environment with a non-blank value."""
    return _raw(name) is not None


def env_bool(name: str, default: bool) -> bool:
    """``name`` parsed as a boolean per this module's truth table.

    Raises:
        ValueError: if set to something outside :data:`TRUE_VALUES` /
            :data:`FALSE_VALUES`.
    """
    raw = _raw(name)
    if raw is None:
        return default
    low = raw.lower()
    if low in _TRUE_SET:
        return True
    if low in _FALSE_SET:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean. Accepted values are "
        f"{'/'.join(TRUE_VALUES)} (true) and {'/'.join(FALSE_VALUES)} (false), "
        f"case-insensitive; unset or empty means the default ({default!r})."
    )


def env_int(name: str, default: int) -> int:
    """``name`` parsed as an ``int``; unset/blank ⇒ ``default``.

    Raises:
        ValueError: if set to something that is not an integer literal.
    """
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not an integer; unset or empty means the default ({default!r})."
        ) from None


def env_float(name: str, default: float) -> float:
    """``name`` parsed as a ``float``; unset/blank ⇒ ``default``.

    Raises:
        ValueError: if set to something that is not a float literal.
    """
    raw = _raw(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a number; unset or empty means the default ({default!r})."
        ) from None


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """``name`` as a plain string; unset/blank ⇒ ``default``.

    For paths and free-form identifiers (store locations, model names) where there
    is no closed value set to validate against.
    """
    raw = _raw(name)
    return default if raw is None else raw


@overload
def env_enum(name: str, default: str, choices: Sequence[str]) -> str: ...


@overload
def env_enum(name: str, default: None, choices: Sequence[str]) -> Optional[str]: ...


def env_enum(
    name: str,
    default: Optional[str],
    choices: Sequence[str],
) -> Optional[str]:
    """``name`` restricted to ``choices`` (case-insensitive); unset/blank ⇒ ``default``.

    The returned value is the *canonical* spelling from ``choices``, so callers can
    compare with ``==``.

    Raises:
        ValueError: if set to a value outside ``choices``.
    """
    raw = _raw(name)
    if raw is None:
        return default
    low = raw.lower()
    for choice in choices:
        if choice.lower() == low:
            return choice
    raise ValueError(
        f"{name}={raw!r} is not one of {', '.join(repr(c) for c in choices)}; "
        f"unset or empty means the default ({default!r})."
    )
