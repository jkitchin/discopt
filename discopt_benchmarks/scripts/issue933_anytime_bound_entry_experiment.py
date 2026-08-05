"""#930 entry experiment: WHY is there no dual bound? -inf root, or tainted tree?

HYPOTHESIS: the 27% of instances that report no dual bound do so because the
tree's `global_lower_bound` is -inf -- the root LP bound was proved but never
installed into node 0 -- NOT because the tree bound was finite and rejected as
tainted. Since `import_results` floors every node at its inherited parent bound
(tree_manager.rs:403) and every child-creation site in branching.rs copies
`local_lower_bound: inherited_lb`, seeding node 0 would propagate to the whole
tree for the whole run.

KILL CRITERION: if the tree bound at the discard point is predominantly FINITE
and rejected for taint, seeding the root is a no-op for this class and the
hypothesis is dead. Sec.4 requires running this BEFORE building the feature.

Sec.6: counts diagnosed cases and exits non-zero if it diagnosed none.
Sec.7: no bare excepts. Sec.8: asserts the #930 marker present.
"""

import json
import os
import subprocess
import sys

BENCH = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark")
NLDIR = os.path.join(BENCH, "minlplib/nl")
NEW = "/Users/jkitchin/projects/discopt"
SCR = os.path.dirname(os.path.abspath(__file__))
TL = float(os.environ.get("E_TL", "8.0"))

INSTS = os.environ.get("E_INSTS", "").split() or [
    "fo8_ar5_1",
    "batchs121208m",
    "portfol_robust100_09",
    "gastrans582_warm15",
    "smallinvDAXr1b020-022",
    "slay10m",
    "crudeoil_lee3_06",
    "squfl030-100",
    "batch0812",
    "rsyn0815m02m",
    "hda",
    "contvar",
]

W = os.path.join(SCR, "_entry_worker.py")
_WORKER_SRC = r'''
import json, sys, math
nl, tl = sys.argv[1], float(sys.argv[2])
import discopt.solver as S
assert hasattr(S, "_admissible_probe_bound"), "MARKER ABSENT: not the #930 tree"

diag = {}

def _capture(tag):
    """Walk up to the solve_model frame and read the decision-point locals."""
    f = sys._getframe(1)
    while f is not None:
        if "_rr_needed" in f.f_locals or "_tree_bound_valid" in f.f_locals:
            L = f.f_locals
            st = L.get("stats")
            glb = None
            if isinstance(st, dict) and "global_lower_bound" in st:
                glb = st["global_lower_bound"]
            diag.update(
                tag=tag,
                tree_glb=(None if glb is None else float(glb)),
                tree_glb_finite=(None if glb is None else bool(math.isfinite(glb))),
                tree_bound_valid=L.get("_tree_bound_valid"),
                gap_certified=L.get("_gap_certified"),
                bound_val_at_decision=(
                    None if L.get("bound_val") is None else float(L["bound_val"])
                ),
                rr_remaining=(
                    None if L.get("_rr_remaining") is None else float(L["_rr_remaining"])
                ),
            )
            return
        f = f.f_back

_rr = S._root_relaxation_lower_bound
def rr(*a, **k):
    _capture("fallback_ran")
    return _rr(*a, **k)
S._root_relaxation_lower_bound = rr

_ap = S._admissible_probe_bound
def ap(*a, **k):
    if "tag" not in diag:
        _capture("probe_direct")
    return _ap(*a, **k)
S._admissible_probe_bound = ap

# The 10-instance "no discard" reading from the first run was an instrument
# no-op (Sec.6): those instances never reach solve_model's _rr_needed block at
# all -- they exit through _solve_nlp_bb / _solve_milp_bb / the native spatial
# kernel, each of which has its own `bound_val = stats["global_lower_bound"]`.
# Hook all of them so "which path" is DATA, not an unlabelled blank.
for _fn in ("_solve_nlp_bb", "_solve_milp_bb", "_try_native_spatial_kernel"):
    _real = getattr(S, _fn)
    def _mk(real, tag):
        def wrapper(*a, **k):
            out = real(*a, **k)
            diag.setdefault("paths", []).append(tag)
            return out
        return wrapper
    setattr(S, _fn, _mk(_real, _fn))

from discopt.modeling.core import from_nl
m = from_nl(nl)
r = m.solve(time_limit=tl)
diag["status"] = r.status
diag["final_bound"] = None if r.bound is None else float(r.bound)
print("RESULT" + json.dumps(diag))
'''
with open(W, "w") as _fh:
    _fh.write(_WORKER_SRC)

env = dict(os.environ)
env["PYTHONPATH"] = os.path.join(NEW, "python")
n_diag = n_inf = n_tainted_finite = n_no_discard = 0
rows = []
print(f"# entry930: {len(INSTS)} instances, TL={TL}s", flush=True)
for name in INSTS:
    nl = os.path.join(NLDIR, name + ".nl")
    if not os.path.exists(nl):
        print(f"{name:26s} MISSING .nl", flush=True)
        continue
    p = subprocess.run(
        [sys.executable, "-u", W, nl, repr(TL)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=NEW,
        env=env,
    )
    r = None
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT"):
            r = json.loads(ln[6:])
    if r is None:
        print(f"{name:26s} NO RESULT: {(p.stderr or '').strip().splitlines()[-1][:70]}", flush=True)
        continue
    rows.append(dict(inst=name, **r))
    if "tag" not in r:
        n_no_discard += 1
        print(
            f"{name:26s} solve_model _rr block NOT REACHED; paths={r.get('paths')} "
            f"final={r['final_bound']}",
            flush=True,
        )
        continue
    n_diag += 1
    if r["tree_glb_finite"] is False:
        n_inf += 1
        verdict = "ROOT NEVER SEEDED (-inf)"
    elif r["tree_bound_valid"] is False:
        n_tainted_finite += 1
        verdict = "FINITE BUT TAINTED  <-- kill criterion"
    else:
        verdict = "finite+valid (other reason)"
    print(
        f"{name:26s} glb={r['tree_glb']!r} valid={r['tree_bound_valid']} "
        f"rr_left={r['rr_remaining']} final={r['final_bound']}  {verdict}",
        flush=True,
    )

print()
print(f"# diagnosed discards      : {n_diag}")
print(f"#   root never seeded (-inf): {n_inf}")
print(f"#   finite but tainted      : {n_tainted_finite}   <-- kills the hypothesis")
print(f"# _rr block not reached   : {n_no_discard}  (instrument silent -- NOT a pass)")
with open(os.path.join(SCR, "entry930.json"), "w") as _fh:
    json.dump(rows, _fh, indent=1)
if n_diag == 0:
    print("PROBE NEVER FIRED: no discard was diagnosed; this run proves nothing", file=sys.stderr)
    sys.exit(1)
