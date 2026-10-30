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
5. :data:`LOCAL_LIMIT` — the sibling of the previous entry: a *point*, from a
   local search that stopped on a limit without establishing stationarity.
   Separate from ``LOCAL_OPTIMAL`` for the same reason ``LOCAL_INFEASIBLE`` is
   separate from ``INFEASIBLE``: the weaker claim needs its own name, or it is
   made under the stronger one.
6. :data:`INFEASIBLE` vs :data:`LOCAL_INFEASIBLE` — a *certified* infeasibility
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

# ── undecided terminations, named by #2053 ──
#
# Both were returned long before they were defined here, so a caller dispatching
# on this module had no branch for them. Neither is a proof of anything, neither
# carries an incumbent, and neither is a failure of the code: the search ended
# and the question it was asked is still open. The benchmark harness maps both
# to its ``UNKNOWN`` bucket, which fails closed.
#: The search ended with no incumbent and no proof either way: feasibility is
#: undetermined. Typical sources: a branch-and-bound tree that drained without a
#: rigorous fathom of every node (so ``infeasible`` would be a false
#: certificate), an external solver whose own status had no discopt meaning
#: (``termination`` then names it), or a decomposition loop (GDPopt LOA, Benders)
#: that stopped with nothing feasible found. Not :data:`ERROR` -- nothing raised
#: or failed -- and not a limit status, because no limit need have been reached.
UNKNOWN = "unknown"
#: The outer-approximation family's spelling of :data:`UNKNOWN` (``solver="oa"``,
#: ``"mip_nlp"``, and the OA-based GDP routes): the master problem ran out of
#: candidate assignments, or stopped, without one NLP subproblem producing a
#: feasible point, AND without the proof that would license ``infeasible`` (a
#: master that is a relaxation and was emptied only by valid rows, #1673 B4).
#: Same claim as ``UNKNOWN`` -- none -- and scored the same way; it is kept as a
#: separate value because callers and tests already match on it.
NO_FEASIBLE_POINT = "no_feasible_point"

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

# ── termination reasons (``SolveResult.termination``), added by #1585 ──
#
# Orthogonal to the status. The status says what was PROVED about the answer;
# the termination reason says why the SEARCH stopped. They differ exactly where
# a caller needs both: a branch-and-bound that hit ``max_nodes`` holding an
# incumbent and a valid bound is ``status="feasible"`` (certified bound +
# incumbent, gap > 0 -- the documented meaning, which existing callers rely on)
# with ``termination="node_limit"``. ``None`` means the route that answered does
# not record a reason (a single local NLP/LP/QP solve, a decomposition loop).
# The pairing below ("gap" -> optimal, "exhausted" -> optimal/infeasible) is the
# usual one, not an invariant: see the ``SolveResult.termination`` comment for
# the known exits where the two disagree.
#: The convergence test (absolute OR relative gap) was met.
TERMINATION_GAP = "gap"
#: Every node of the tree was fathomed or proved; the search ran to completion.
TERMINATION_EXHAUSTED = "exhausted"
#: The tree drained, but a subtree was removed WITHOUT proof -- the #467 pin (no
#: branch direction and no finite bound, typically an unbounded dimension), a
#: non-rigorous sentinel fathom, or a #598 unresolved floor that keeps the gap
#: open -- so the search did not run to completion and ``"exhausted"`` would
#: overstate it (#1869). ``status`` decides what was still proven.
TERMINATION_UNRESOLVED = "unresolved"
#: ``max_nodes`` was reached with nodes still open.
TERMINATION_NODE_LIMIT = "node_limit"
#: ``time_limit`` was reached (including the tail reserved for the root-relaxation
#: bound fallback).
TERMINATION_TIME_LIMIT = "time_limit"
#: An iteration cap stopped the solve.
TERMINATION_ITERATION_LIMIT = "iteration_limit"
#: The user stopped the search (the interactive debugger's ``quit``).
TERMINATION_INTERRUPTED = "interrupted"
#: A local NLP solve stopped on Ipopt/POUNCE's ``Diverging_Iterates``: the iterate
#: norm ran past the divergence threshold. That *suggests* an objective unbounded
#: on the feasible set but proves nothing (#2053); the iterate, when there is
#: one, is in ``last_iterate`` and never in ``x``.
TERMINATION_DIVERGING = "diverging_iterates"
#: Presolve (FBBT / OBBT / nonlinear bound tightening) proved the model
#: infeasible before any search ran; ``status`` is ``infeasible``, ``node_count``
#: is 0 and ``algorithm_route`` names the proving pass (#1618 C-17).
TERMINATION_PRESOLVE = "presolve"

#: Every value ``SolveResult.termination`` may take besides ``None``.
TERMINATION_REASONS = frozenset(
    {
        TERMINATION_GAP,
        TERMINATION_EXHAUSTED,
        TERMINATION_UNRESOLVED,
        TERMINATION_NODE_LIMIT,
        TERMINATION_TIME_LIMIT,
        TERMINATION_ITERATION_LIMIT,
        TERMINATION_INTERRUPTED,
        TERMINATION_PRESOLVE,
        TERMINATION_DIVERGING,
    }
)

#: Every value ``SolveResult.status`` may take (#2053). A status string a route
#: returns that is not in this set is a defect: ``test_2053_status_vocabulary``
#: scans the package for status literals and fails on one.
STATUSES = frozenset(
    {
        OPTIMAL,
        FEASIBLE,
        INFEASIBLE,
        UNBOUNDED,
        TIME_LIMIT,
        NODE_LIMIT,
        ITERATION_LIMIT,
        ERROR,
        UNKNOWN,
        NO_FEASIBLE_POINT,
        LOCAL_OPTIMAL,
        LOCAL_LIMIT,
        LOCAL_INFEASIBLE,
    }
)

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
