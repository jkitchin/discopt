"""G2 — the effort governor: hit-rate-adaptive root-heuristic scheduling.

Motivation (measured, ``docs/dev/g2-effort-governor-2026-07-07.md``): the root
primal-heuristic phase is 60-97 % of easy-instance wall, and the volume is
largely *non-load-bearing*. On a pooled panel + held-out replay (27 instances,
60 s, gap 1e-4) the incumbent came from ``_solve_root_node_multistart`` (start
#1) on **every** instance, while the expensive nested-B&B heuristics never
improved it:

============================  ======  ========  ===========
source                        solves  wall%     incumbent hit-rate
============================  ======  ========  ===========
rens (nested B&B)                805     14.3 %   0 / 13   (0.0 %)
rins (LNS dive)                  440     10.4 %   0 / 143  (0.0 %)
============================  ======  ========  ===========

On the 21 *finished* easy instances RENS alone is **33 % of total solve wall**
at a 0 % hit rate. SCIP/BARON track per-heuristic success and throttle the
losers; discopt fires the full suite unconditionally at every root. This module
is the missing conductor.

Policy (global constants, documented, NOT keyed on instance name or shape —
CLAUDE.md §2):

* Cheap *finder* heuristics (multistart, pump, diving) are **never** governed:
  securing the first incumbent for pruning always wins, and they are cheap.
* Each governed (expensive) source is tracked by a **process-lifetime class
  hit-rate** (``rens``, ``rins``, ``lbranch``, ``enumerate`` are *classes*, not
  instances). A source that fails to improve the incumbent ``K_DISABLE``
  consecutive times is disabled for the rest of the process. The cross-*solve*
  memory is deliberate: RENS fires once per solve, so a per-solve counter could
  never throttle it — the only way to throttle a loser that fires once per solve
  is to remember it lost on the last solves, which is exactly how SCIP/BARON
  behave across a run.
* Expensive sources additionally run **only while the primal-dual gap is open**
  (a closed gap means the certificate is essentially done — no improver can
  help).

Soundness (heuristic-policy regime, CLAUDE.md §5): throttling a primal heuristic
can only ever cost B&B *nodes* — never a wrong optimum, bound, or lost
certificate. B&B stays exhaustive; the governor only decides whether to *spend*
effort looking for a better incumbent, and every injected point is re-verified
downstream. The dual bound is never touched.

Default-**OFF**: the governor is inert unless ``DISCOPT_HEURISTIC_GOVERNOR`` is
set to a truthy value (``1``/``true``/``on``). With it off, :meth:`allowed`
returns ``True`` for every source and the solve path is byte-identical to the
pre-governor behaviour. Graduation to default-ON goes through the G1.2 flag
gate, not this module.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

# ---- global policy constants (NOT per-instance / per-name) -------------------

#: A governed source disabled after this many *consecutive* class misses
#: (calls that did not strictly improve the incumbent). Small: the measured
#: hit rate of the governed sources is 0 %, so one or two misses is already
#: conclusive, and a load-bearing source that hits resets its miss streak.
K_DISABLE = 2

#: Sources the governor treats as *expensive* — they additionally require the
#: primal-dual gap to be open.
#:
#: SCOPE (measurement-bounded, CLAUDE.md §4 — the measurement wins). The entry
#: experiment found rens AND rins at a 0 % pooled incumbent hit-rate on the
#: easy panel + held-out MINLP slice, but the cert panel then showed **rins /
#: local-branching are load-bearing on the convex-nonseparable class**
#: (cvxnonsep_psig30: throttling them lost the better incumbent, 78.9989 ->
#: 79.0024, degrading the certified objective at the gap tolerance). So the
#: governed set is narrowed to the source whose 0 %-hit generalizes across every
#: class tested AND that carries the dominant wall: **rens** (33 % of finished
#: easy-class solve wall; a whole nested B&B fired once at the root, at a 0 % hit
#: rate on every instance INCLUDING the convex ones, where multistart already
#: has the incumbent). RINS, local-branching and the binary-seed enumeration are
#: left to the existing ``_improver_allowed`` node-budget contingent — they are
#: NOT governed here, precisely because the evidence for throttling them does not
#: hold on the convex class.
EXPENSIVE_SOURCES = frozenset({"rens"})

#: The sources the governor throttles at all. A source not listed here is always
#: allowed and never accrues a throttle (``record`` still tracks its stats for
#: observability, but never disables it). Currently identical to
#: :data:`EXPENSIVE_SOURCES`; kept separate so a future cheap-but-losing source
#: could be governed without the gap gate.
GOVERNED_SOURCES = frozenset({"rens"})


def _governor_enabled() -> bool:
    return os.environ.get("DISCOPT_HEURISTIC_GOVERNOR", "0").lower() in ("1", "true", "on", "yes")


@dataclass
class _SourceStats:
    calls: int = 0
    hits: int = 0
    consecutive_misses: int = 0
    disabled: bool = False
    throttled_events: int = 0  # times :meth:`allowed` refused this source


@dataclass
class HeuristicGovernor:
    """Per-source running stats + the throttle policy.

    Thread-safe (a single :data:`_LOCK` guards the mutable stats) so it can be a
    process-lifetime singleton shared across solves. Cheap: two dict lookups per
    heuristic call site.
    """

    stats: dict[str, _SourceStats] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _get(self, source: str) -> _SourceStats:
        st = self.stats.get(source)
        if st is None:
            st = _SourceStats()
            self.stats[source] = st
        return st

    def allowed(self, source: str, *, gap_open: bool = True) -> bool:
        """Whether a governed heuristic ``source`` may run now.

        ``gap_open`` is honoured only for :data:`EXPENSIVE_SOURCES`. Returns
        ``True`` unconditionally when the governor is disabled (default) so the
        caller's behaviour is unchanged.
        """
        if not _governor_enabled() or source not in GOVERNED_SOURCES:
            return True
        with self._lock:
            st = self._get(source)
            if st.disabled:
                st.throttled_events += 1
                return False
            if source in EXPENSIVE_SOURCES and not gap_open:
                st.throttled_events += 1
                return False
            return True

    def record(self, source: str, improved: bool) -> None:
        """Charge a governed run against the class hit-rate.

        ``improved`` is ``True`` iff the run strictly improved the incumbent.
        A miss streak of :data:`K_DISABLE` disables the source for the rest of
        the process; any hit resets the streak (a source that starts paying off
        again is re-enabled by :meth:`record` clearing ``disabled``).
        """
        if not _governor_enabled() or source not in GOVERNED_SOURCES:
            return
        with self._lock:
            st = self._get(source)
            st.calls += 1
            if improved:
                st.hits += 1
                st.consecutive_misses = 0
                st.disabled = False
            else:
                st.consecutive_misses += 1
                if st.consecutive_misses >= K_DISABLE:
                    st.disabled = True

    # -- introspection (the firing proof / instrumentation) -------------------

    def snapshot(self) -> dict[str, dict[str, object]]:
        """A JSON-friendly copy of the per-source stats (for the firing proof)."""
        with self._lock:
            return {
                src: {
                    "calls": st.calls,
                    "hits": st.hits,
                    "consecutive_misses": st.consecutive_misses,
                    "disabled": st.disabled,
                    "throttled_events": st.throttled_events,
                }
                for src, st in self.stats.items()
            }

    def any_throttled(self) -> bool:
        """True iff the governor refused at least one governed source."""
        with self._lock:
            return any(st.throttled_events > 0 for st in self.stats.values())

    def reset(self) -> None:
        """Clear all class stats (test isolation / a fresh benchmark run)."""
        with self._lock:
            self.stats.clear()


#: Process-lifetime singleton. Governed call sites route through this.
_GOVERNOR = HeuristicGovernor()


def governor() -> HeuristicGovernor:
    """Return the process-lifetime governor singleton."""
    return _GOVERNOR
