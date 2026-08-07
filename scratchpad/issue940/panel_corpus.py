"""Issue #940 corpus panel: bound-neutrality over the in-repo .nl corpus.

Per CLAUDE.md §5 regime 1, a change that should not move the search must be shown
not to. The POUNCE LP+QP change is expected to be inert on this corpus (the
per-node engine is the Rust simplex, and 0 of these 68 instances classify as a
pure LP), so anything but an exact status/objective/bound/node_count match is a
finding.

Both arms run INTERLEAVED in one process (CLAUDE.md §9) — the arm is selected by
flipping the module globals the backends read, not by two sequential runs against
two checkouts — so background load cannot land on one arm only. The pre-fix arm is
reconstructed exactly: POUNCE's own 1e-4 ``constr_viol_tol`` and no compact-box
refusal.

§6: prints a comparison count and exits non-zero when it is zero.
"""

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solvers.lp_pounce as LPP  # noqa: E402
import discopt.solvers.qp_pounce as QPP  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

# §8: assert we loaded the code under test, by file AND by a marker unique to it.
assert LPP.__file__.startswith("/home/user/discopt/python/"), LPP.__file__
assert hasattr(LPP, "_reject_impossible_unbounded"), "post-fix marker absent"
assert LPP._CONSTR_VIOL_TOL == 1e-8, LPP._CONSTR_VIOL_TOL

_POST_TOL = LPP._CONSTR_VIOL_TOL
_POST_REJECT = LPP._reject_impossible_unbounded
_POST_COMPACT = QPP._box_is_compact
_IPOPT_DEFAULT_TOL = 1e-4

ROOT = "python/tests/data/minlplib_nl"
TIME_LIMIT = 20.0


def set_arm(arm):
    """Select 'pre' (POUNCE's own defaults) or 'post' (the #940 backends)."""
    if arm == "pre":
        LPP._CONSTR_VIOL_TOL = _IPOPT_DEFAULT_TOL
        QPP._CONSTR_VIOL_TOL = _IPOPT_DEFAULT_TOL
        LPP._reject_impossible_unbounded = lambda result, lb, ub: result
        QPP._box_is_compact = lambda lb, ub: False
    else:
        LPP._CONSTR_VIOL_TOL = _POST_TOL
        QPP._CONSTR_VIOL_TOL = _POST_TOL
        LPP._reject_impossible_unbounded = _POST_REJECT
        QPP._box_is_compact = _POST_COMPACT


def solve_one(path, arm):
    set_arm(arm)
    t0 = time.perf_counter()
    res = from_nl(path).solve(time_limit=TIME_LIMIT)
    return {
        "status": res.status,
        "objective": res.objective,
        "bound": res.bound,
        "node_count": res.node_count,
        "wall": time.perf_counter() - t0,
    }


def main(out_path):
    files = sorted(f for f in os.listdir(ROOT) if f.endswith(".nl"))
    rows = {}
    comparisons = 0
    diffs = []
    for i, f in enumerate(files, 1):
        name = f[:-3]
        p = os.path.join(ROOT, f)
        # Alternate which arm runs first so a warm-cache or drift effect cannot
        # systematically favour one of them.
        order = ("pre", "post") if i % 2 else ("post", "pre")
        got = {a: solve_one(p, a) for a in order}
        rows[name] = got
        comparisons += 1
        keys = ("status", "objective", "bound", "node_count")
        same = all(got["pre"][k] == got["post"][k] for k in keys)
        if not same:
            diffs.append((name, {k: (got["pre"][k], got["post"][k]) for k in keys
                                 if got["pre"][k] != got["post"][k]}))
        print(f"[{i:3d}/{len(files)}] {name:28s} {'SAME' if same else 'DIFF'} "
              f"pre={got['pre']['status']}/{got['pre']['objective']!r}/"
              f"n{got['pre']['node_count']} post={got['post']['status']}/"
              f"{got['post']['objective']!r}/n{got['post']['node_count']}", flush=True)

    json.dump({"rows": rows, "diffs": diffs}, open(out_path, "w"), indent=1)
    print(f"\nCOMPARISONS_EXECUTED={comparisons}  identical={comparisons - len(diffs)}  "
          f"differing={len(diffs)}")
    for name, d in diffs:
        print(f"  DIFF {name}: {d}")
    if comparisons == 0:
        print("PANEL COMPARED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
