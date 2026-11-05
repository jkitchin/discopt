"""Scope POUNCE's infeasibility retry ladder to the NLPs whose verdict it can change.

When a solve ends "locally infeasible", POUNCE re-solves it up to five more ways
(scaling, ``mu`` strategy, perturbed start, saddle escape, a higher-quality
factorization) before it reports that. The second opinion is worth paying for
only where an "infeasible" verdict is acted on as if it were a fact. Elsewhere
it buys a point that is merely more likely.

Where a verdict is not a proof:

* **Primal heuristics.** A failed sub-NLP loses one candidate incumbent and
  nothing else.
  The scope covers everything a heuristic runs, including the nested
  ``solve_model`` that RINS and RENS run on a restricted sub-model (convex nodes
  and all), whose only output is a proposed incumbent.
* **Node NLPs on a nonconvex model.** "Locally infeasible" is not an
  infeasibility proof there (``_solve_node_nlp_pounce``), and the node stays open.
  Since #1537 a node is pruned on a POUNCE infeasible label only when a Farkas
  certificate on its linear system confirms it.

Everywhere else the ladder stays on: node NLPs on convex models, the
single-NLP route, OA, LOA, AMP, Benders and every other caller.

Measured on GDPlib under load 6–8, ladder on vs off for every serial NLP, with
identical objectives, bounds and certificates in every arm:

| instance | on | off |
|---|---|---|
| clay0203 big-m | 38.8 s | 13.8 s |
| ex_633 hull | 7.9 s | 5.0 s |
| nonconvex_hen hull | 7.7 s | 6.0 s |
| syngas big-m | 116 s | 104 s |

Five other instances showed no difference. This module turns the ladder off
only inside the two scopes above.

**Flag.** ``DISCOPT_NLP_LADDER_SCOPED=1``, default OFF (CLAUDE.md §5). The change
alters which incumbents are found and when, so it changes node counts. It
graduates only on a corpus panel that is cert-clean and net-positive. The row
is in ``docs/dev/flag-retirement-audit.md``.

**Precedence.** A retry option the caller set explicitly (``pounce_options``)
always wins. The scope only fills defaults.
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from typing import Iterator

#: POUNCE's infeasibility second-opinion retries (``pounce`` 0.15).
LADDER_RETRY_OPTIONS: tuple[str, ...] = (
    "feral_infeasibility_scaling_retry",
    "infeasibility_mu_strategy_retry",
    "infeasibility_perturbed_start_retry",
    "infeasibility_saddle_escape_retry",
    "feral_increase_quality_retry",
)

#: True inside a scope whose NLP verdicts are not acted on as proofs.
LADDER_OFF: ContextVar[bool] = ContextVar("_NLP_LADDER_OFF", default=False)


def ladder_scope_enabled() -> bool:
    """``DISCOPT_NLP_LADDER_SCOPED`` (default OFF), read at call time."""
    return os.environ.get("DISCOPT_NLP_LADDER_SCOPED", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


@contextlib.contextmanager
def ladder_off(active: bool = True) -> Iterator[None]:
    """Mark this block's POUNCE solves as not needing the retry ladder.

    A no-op when the flag is off or ``active`` is False.
    """
    if not active or not ladder_scope_enabled():
        yield
        return
    token = LADDER_OFF.set(True)
    try:
        yield
    finally:
        LADDER_OFF.reset(token)


def apply_ladder_scope(opts: dict) -> dict:
    """Set the retry options to ``"no"`` in ``opts`` when inside :func:`ladder_off`.

    Explicit caller settings are kept. Returns ``opts``, modified in place.
    """
    if LADDER_OFF.get():
        for key in LADDER_RETRY_OPTIONS:
            opts.setdefault(key, "no")
    return opts
