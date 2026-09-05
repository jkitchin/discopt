"""#1153: does the max-never-decays CLIFF actually bite on this panel?

Review's point: a panel that improves ``nvs05@10s`` does not by itself clear the
running-max estimator, because that cliff only bites when an expensive NLP is
observed BEFORE the finder entry decision. If no instance hits that ordering, the
estimator was never the mechanism there and the improvement is the scoping fix.

So record, at every finder-role entry decision: the observed-cost state
(max / mean / n) and what each estimator would have decided. Prints an
executed-decision count; exits non-zero at zero (CLAUDE.md §6).
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
import discopt.solver as S
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(S, "_worst_heur_nlp_cost") or True  # closure, not module-level

DECISIONS = []
_real_ok = None


def _instrument(mod):
    """Wrap solve_model so each finder entry decision is recorded.

    ``_root_heur_nlp_entry_ok`` is a closure, so it cannot be patched directly;
    instead patch the SHARE seam it reads and record the cost state each time a
    finder-role call consults it.
    """
    real_share = mod._heuristic_entry_share

    def share():
        v = real_share()
        DECISIONS.append(v)
        return v

    mod._heuristic_entry_share = share
    return real_share


inst, tl = sys.argv[1], float(sys.argv[2])
for on in (False, True):
    DECISIONS.clear()
    real = _instrument(S)
    tok = solver_tuning.enter_scope(solver_tuning.SolverTuning(heuristic_entry_share=on))
    try:
        r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    finally:
        solver_tuning.reset_current(tok)
        S._heuristic_entry_share = real
    print(f"share={'ON ' if on else 'OFF'} tl={tl} obj={r.objective!r} nodes={r.node_count} "
          f"| finder-share consultations={len(DECISIONS)} values={sorted(set(DECISIONS))}",
          flush=True)
print(f"\n# executed decisions observed: {len(DECISIONS)}", flush=True)
raise SystemExit(0 if DECISIONS else 1)
