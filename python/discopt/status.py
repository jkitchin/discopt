"""The terminal-status vocabulary, and what each member is allowed to claim.

A solve's *status* is the first thing every downstream consumer reads — the
release gate, the benchmark harness, a panel script, a caller deciding whether
to trust an objective. Before #1148 the vocabulary had no member meaning
**"local stationary point, no global claim"**, so a continuous local mode had
only two options and both were wrong:

* reuse ``"optimal"`` — and the benchmark harness scores the row as a *proved
  optimum* (``metrics.is_solved`` is ``status == OPTIMAL``, and
  ``proved_optimal_count`` counts ``gap is None and is_solved`` as proved), so
  a local stationary point is read as a certificate. That is CLAUDE.md §1's
  hard gate, with no slack;
* add an optional ``certified=False`` field beside an unchanged ``"optimal"``
  — which fixes nothing, because every consumer that pattern-matches on the
  *status* inherits the bug.

So the distinct status is the mechanism. An unknown status string maps to
``UNKNOWN`` in the harness, ``is_solved`` is False, and both ``incorrect_count``
and ``proved_optimal_count`` skip the row: a consumer that has never heard of
the local mode **fails closed**. ``gap_certified`` stays a refinement of a
status, never the discriminator.

The states this vocabulary keeps apart
--------------------------------------

1. :data:`OPTIMAL` — certified optimal.
2. :data:`FEASIBLE` (with a finite ``bound``) — certified bound + incumbent, gap > 0.
3. :data:`FEASIBLE` (no ``bound``) — feasible, no bound.
4. :data:`LOCAL_OPTIMAL` — local stationary point, **no global claim**.
4b. :data:`LOCAL_LIMIT` — a *point*, from a local search that stopped on a
   limit without establishing stationarity. Separate from ``LOCAL_OPTIMAL``
   for the same reason ``LOCAL_INFEASIBLE`` is separate from ``INFEASIBLE``:
   the weaker claim needs its own name, or it is made under the stronger one.
5. :data:`INFEASIBLE` vs :data:`LOCAL_INFEASIBLE` — a *certified* infeasibility
   proof versus a local solver that merely failed to find a point. A stalled
   MPEC continuation must never surface as ``"infeasible"``: that is a false
   certificate in the other direction, and it is why the two have separate
   names rather than a shared one plus a flag.
"""

from __future__ import annotations

# ── certified terminal statuses ──
#: Certified global optimum: the reported gap is proved.
OPTIMAL = "optimal"
#: A feasible incumbent; ``bound`` says whether a certified dual bound came with it.
FEASIBLE = "feasible"
#: **Certified** infeasibility — a proof, not a failed search.
INFEASIBLE = "infeasible"
#: Certified unboundedness.
UNBOUNDED = "unbounded"

# ── limit terminations ──
TIME_LIMIT = "time_limit"
NODE_LIMIT = "node_limit"
ITERATION_LIMIT = "iteration_limit"

# ── failures ──
ERROR = "error"

# ── local (no global claim) terminations, added by #1148 ──
#: A local stationary point of the problem actually solved. Carries **no**
#: global claim: it may become an incumbent after independent feasibility
#: verification, and may never become a dual bound.
LOCAL_OPTIMAL = "local_optimal"
#: A local solver produced a point but never established stationarity: it
#: stopped on an iteration or step-size limit at an *iterate*. Strictly weaker
#: than :data:`LOCAL_OPTIMAL`, and kept apart from it because ``local_optimal``
#: is defined above as "a local stationary point" and a consumer is entitled to
#: read it that way. A Scholtes homotopy whose subsolver did zero iterations
#: returned its own starting point under ``local_optimal`` (#1158 review 3,
#: blocking 3); the point is still worth reporting — as a warm start, and with
#: its residuals — but not under a label that claims more than happened.
LOCAL_LIMIT = "local_limit"
#: A local solver failed to produce a point. This is *not* an infeasibility
#: proof — a stalled MPEC continuation lands here, never on :data:`INFEASIBLE`.
LOCAL_INFEASIBLE = "local_infeasible"

#: Statuses that make no global claim. A result carrying one of these is never
#: certified and never contributes a dual bound.
LOCAL_STATUSES = frozenset({LOCAL_OPTIMAL, LOCAL_LIMIT, LOCAL_INFEASIBLE})

#: Statuses that assert a proof about the *global* problem.
CERTIFYING_STATUSES = frozenset({OPTIMAL, INFEASIBLE, UNBOUNDED})


def is_local_status(status: object) -> bool:
    """True when ``status`` makes no global claim (see :data:`LOCAL_STATUSES`).

    Takes ``object`` rather than ``str`` on purpose: ``SolveResult.status`` is
    typed ``str`` but reaches this from JSON, from an enum's ``.value``, and
    from adapters, and a non-string must answer "not local" rather than raise
    inside a soundness guard.
    """
    return isinstance(status, str) and status in LOCAL_STATUSES


def is_certified_status(status: object) -> bool:
    """True when ``status`` asserts a proof about the global problem.

    Absent or unrecognised certification is interpreted as **not certified**
    (`#1148 <https://github.com/jkitchin/discopt/issues/1148>`_ §C), which is
    why this is a membership test against a closed set rather than
    ``not is_local_status(...)``.
    """
    return isinstance(status, str) and status in CERTIFYING_STATUSES
