"""#1153: does the max-never-decays CLIFF actually bite on this panel?

Review's point: a panel that improves ``nvs05@10s`` does not by itself clear the
running-max estimator, because that cliff only bites when an expensive NLP is
observed BEFORE the finder entry decision. If the finder decision happens while
no NLP has been observed at all, ``max == mean == the 2.0 s default`` and the two
estimators are numerically IDENTICAL — so the estimator cannot be the mechanism
there, whatever the outcome.

This records the interleaved sequence of (heuristic NLP observed, wall) and
(finder share consulted) events, which answers the ordering question directly
instead of inferring it from the result. Executed-event count; non-zero at zero.
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
import discopt.solver as S
import discopt.solvers.nlp_pounce as NP
from discopt import solver_tuning
from discopt.modeling.core import from_nl

print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
assert hasattr(solver_tuning, "heuristic_entry_share"), "marker absent — wrong tree"

EVENTS = []
_real_nlp = NP.solve_nlp
_real_share = S._heuristic_entry_share


def _nlp(evaluator, x0, **kw):
    t = time.perf_counter()
    try:
        return _real_nlp(evaluator, x0, **kw)
    finally:
        EVENTS.append(("nlp", time.perf_counter() - t))


def _share():
    v = _real_share()
    EVENTS.append(("finder-decision", v))
    return v


inst, tl = sys.argv[1], float(sys.argv[2])
n_events = 0
for on in (False, True):
    EVENTS.clear()
    NP.solve_nlp, S._heuristic_entry_share = _nlp, _share
    tok = solver_tuning.enter_scope(solver_tuning.SolverTuning(heuristic_entry_share=on))
    try:
        r = from_nl(inst).solve(time_limit=tl, gap_tolerance=1e-4)
    finally:
        solver_tuning.reset_current(tok)
        NP.solve_nlp, S._heuristic_entry_share = _real_nlp, _real_share
    n_events += len(EVENTS)
    seq = list(EVENTS)
    first = next((i for i, e in enumerate(seq) if e[0] == "finder-decision"), None)
    prior = [w for k, w in seq[:first or 0] if k == "nlp"]
    print(f"\nshare={'ON ' if on else 'OFF'} tl={tl} obj={r.objective!r} nodes={r.node_count}",
          flush=True)
    print(f"  events: {len(seq)}  first finder decision at index {first}", flush=True)
    print(f"  heuristic NLPs observed BEFORE that decision: {len(prior)}"
          f"{' walls=' + str([round(w, 2) for w in prior]) if prior else ''}", flush=True)
    if not prior:
        print("  -> max == mean == the 2.0 s default at the decision: the two "
              "estimators are IDENTICAL here, so the estimator cannot be the "
              "mechanism on this instance.", flush=True)
    else:
        print(f"  -> max={max(prior):.2f}s vs mean={sum(prior) / len(prior):.2f}s "
              "at the decision: the estimators differ, the cliff can bite.", flush=True)
print(f"\n# executed events: {n_events}", flush=True)
raise SystemExit(0 if n_events else 1)
