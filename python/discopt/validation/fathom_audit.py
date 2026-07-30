"""A default-inactive audit hook for the *bound-fathom* decision.

Why this module exists
----------------------
Card 4c set out to port three stray branch-and-bound loops
(``_jax/lp_spatial_bb``, ``gp/solve_gp_minlp``,
``_jax/convexity/signomial_global``) onto ``PyTreeManager``, on the rationale
that "every certificate-critical pruning decision now flows through one audited
tree manager". Two entry experiments (plan §6, 2026-07-30) killed the port: the
Regime-N panel cannot invoke two of the three loops at all, the only
budget-independent comparable is a five-node tree, and a faithful port would have
to add five policy switches plus two contract extensions to the audited
component — making it *harder* to audit, not easier.

The **goal** the port served is still worth serving, and it does not require the
port: what makes a pruning decision auditable is that it is *observable*, not
that it is centralised. This module is that observation point. Each loop reports
every bound-fathom decision — both arms, kept and fathomed — through
:func:`record_fathom`; a test (or any caller) installs a hook with
:func:`fathom_audit` and re-derives the invariant independently.

The invariant under audit
-------------------------
**A node is never fathomed while its bound is better than the incumbent by more
than the optimality tolerance.** Fathoming such a node discards a subtree that
could still contain the optimum — a false ``optimal`` certificate, the exact
failure class CLAUDE.md §1 puts above everything else.

Note the direction that matters: the check must be recomputed by the auditor from
the solve's *declared* ``gap_tolerance``, never from the slack the loop itself
used. Asserting ``node_bound >= incumbent - loop_reported_slack`` is a tautology
— the loop only fathoms when that holds — and would read as a pass while
measuring nothing (CLAUDE.md §6). Records therefore carry the loop's own
``slack`` for diagnosis but the auditor is expected to ignore it when judging.

Sense convention
----------------
``node_bound`` and ``incumbent`` are always recorded in the loop's **internal
minimisation** sense, so ``incumbent - node_bound`` is the (signed) improvement
still available in that subtree, whatever the model's declared sense. Callers in
a maximise model must map through their own ``to_internal`` before recording.

Cost when inactive
------------------
One module-global read and an ``is None`` test per decision. No allocation, no
formatting, no import. The hook is process-global and not thread-safe by design:
it is an instrument, and a shared mutable hook is what lets a test observe a
solve it does not own.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

__all__ = [
    "FathomDecision",
    "FathomLog",
    "fathom_audit",
    "get_fathom_hook",
    "record_fathom",
    "set_fathom_hook",
]


@dataclass(frozen=True)
class FathomDecision:
    """One bound-fathom decision, kept or fathomed.

    Attributes
    ----------
    loop
        Which loop made the decision (``"gp_minlp"``, ``"signomial_global"``,
        ``"lp_spatial_bb"``).
    site
        A stable label for the decision point within that loop, so a violation
        localises to a line rather than to a file.
    node_bound
        The node's rigorous lower bound, **internal minimisation sense**.
    incumbent
        The incumbent objective at the moment of the decision, same sense.
        ``+inf`` when there is none.
    fathomed
        True if the loop discarded (or stopped exploring) the node here.
    slack
        The slack the loop itself applied. Recorded for diagnosis only — an
        auditor judging the invariant must re-derive its own admissible slack
        from the declared ``gap_tolerance`` (see the module docstring).
    extra
        Free-form loop-specific context (node index, box width, …).
    """

    loop: str
    site: str
    node_bound: float
    incumbent: float
    fathomed: bool
    slack: float
    extra: dict[str, Any] = field(default_factory=dict)

    def improvement(self) -> float:
        """``incumbent - node_bound``: how much this subtree could still gain.

        Positive means the node claims to be *better* than the incumbent. A
        fathomed record with an improvement exceeding the admissible tolerance is
        the violation this module exists to catch.
        """
        return self.incumbent - self.node_bound


_HOOK: Optional[Callable[[FathomDecision], None]] = None


def get_fathom_hook() -> Optional[Callable[[FathomDecision], None]]:
    """Return the installed hook, or None when auditing is inactive."""
    return _HOOK


def set_fathom_hook(
    hook: Optional[Callable[[FathomDecision], None]],
) -> Optional[Callable[[FathomDecision], None]]:
    """Install *hook* (or None to disable). Returns the previous hook."""
    global _HOOK
    previous = _HOOK
    _HOOK = hook
    return previous


def record_fathom(
    loop: str,
    site: str,
    *,
    node_bound: float,
    incumbent: float,
    fathomed: bool,
    slack: float,
    **extra: Any,
) -> None:
    """Report one bound-fathom decision to the installed hook.

    A no-op — two bytecode-cheap operations — when no hook is installed.

    Exceptions raised by the hook are **not** caught (CLAUDE.md §7): an
    instrument that swallows its own failure turns "this path is broken" into
    "this path is fine".
    """
    hook = _HOOK
    if hook is None:
        return
    hook(
        FathomDecision(
            loop=loop,
            site=site,
            node_bound=float(node_bound),
            incumbent=float(incumbent),
            fathomed=bool(fathomed),
            slack=float(slack),
            extra=extra,
        )
    )


class FathomLog(list):
    """The decisions collected by one :func:`fathom_audit` block.

    A plain ``list`` of :class:`FathomDecision` with two accessors that keep the
    executed-count discipline (CLAUDE.md §6) at the call site short.
    """

    def fathomed(self) -> list[FathomDecision]:
        """Only the decisions that actually discarded a node."""
        return [d for d in self if d.fathomed]

    def for_loop(self, loop: str) -> "FathomLog":
        """The subset produced by one loop."""
        out = FathomLog()
        out.extend(d for d in self if d.loop == loop)
        return out


@contextmanager
def fathom_audit() -> Iterator[FathomLog]:
    """Collect every bound-fathom decision made inside the block.

    Example
    -------
    >>> with fathom_audit() as log:          # doctest: +SKIP
    ...     result = model.solve(solver="gp-minlp")
    >>> len(log) > 0                         # doctest: +SKIP
    True

    The previous hook is restored on exit, including on an exception, so a nested
    or concurrent audit cannot silently lose its own records.
    """
    log = FathomLog()
    previous = set_fathom_hook(log.append)
    try:
        yield log
    finally:
        set_fathom_hook(previous)
