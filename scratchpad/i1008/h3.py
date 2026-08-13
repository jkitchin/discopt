"""#1008 H3 entry experiment: WHICH HALF of the refactorization is the 59.5%?

The entry flamegraph attributed 59.5% of wall to "refactorize". A sparse
refactorization is two very different things:

  * SYMBOLIC (`SparseLuSymbolic::analyze`) — AMD on the AtA pattern. discopt
    recomputes it from scratch on every refactorization even though an LP basis
    changes by exactly one column per pivot. Cost scales with nnz(AtA), which for
    a basis containing one moderately dense row is quadratic in that row's length.
  * NUMERIC (`SparseLu::factor`) — the elimination itself. Cost scales with
    nnz(L+U), which the headroom experiment just proved is already at parity with
    SuperLU and therefore has no headroom.

These have completely different fixes, so the split decides the whole issue.

Reads the per-solve `profile::dump()` (stderr) which now carries `LuSymbolic` and
`LuNumeric`. Prints a parsed-solve count and exits non-zero at zero (CLAUDE.md
#6). Nothing is caught (#7).
"""

import glob
import json
import os
import re
import subprocess
import sys

os.environ["DISCOPT_PROFILE"] = "1"

import numpy as np
import scipy.sparse as sp

import discopt
import discopt._rust as _rust

WT = "/Users/jkitchin/projects/discopt/.claude/worktrees/agent-a21bb4a7ae1704077"
assert discopt.__file__.startswith(WT), discopt.__file__
assert _rust.__file__.startswith(WT), _rust.__file__
strs = subprocess.run(
    f"strings {_rust.__file__}", shell=True, capture_output=True, text=True, check=True
).stdout
assert "LuSymbolic" in strs, "loaded a build without the #1008 LU phase split"

from discopt.solvers.milp_simplex import _dual_start_slack_basis

PHASE = re.compile(r"^\s+(\w+)\s+(\d+) calls\s+([\d.]+) ms\s*$")


def run_one(path, tl):
    """Solve one captured LP in a CHILD process so its stderr dump is isolated."""
    code = (
        "import os,sys,json;os.environ['DISCOPT_PROFILE']='1';"
        "sys.path.insert(0,%r);" % (WT + "/python")
        + "import numpy as np,scipy.sparse as sp,discopt._rust as _rust;"
        "from discopt.solvers.milp_simplex import _dual_start_slack_basis;"
        "z=np.load(%r);" % path
        + "nrow,ncol=int(z['shape'][0]),int(z['shape'][1]);"
        "A=sp.csc_matrix((z['data'],z['indices'],z['indptr']),shape=(nrow,ncol));"
        "c,b,lo,hi=z['c'],z['b'],z['lo'],z['hi'];"
        "st=_dual_start_slack_basis(c,lo,hi,nrow);"
        "assert st is not None;"
        "af=sp.hstack([A,sp.identity(nrow,format='csc')],format='csc');"
        "out=_rust.solve_lp_warm_csc_py("
        "np.ascontiguousarray(np.concatenate([c,np.zeros(nrow)])),nrow,ncol+nrow,"
        "np.ascontiguousarray(af.indptr,dtype=np.int64),"
        "np.ascontiguousarray(af.indices,dtype=np.int64),"
        "np.ascontiguousarray(af.data,dtype=np.float64),"
        "np.ascontiguousarray(b),"
        "np.ascontiguousarray(np.concatenate([lo,np.zeros(nrow)])),"
        "np.ascontiguousarray(np.concatenate([hi,np.full(nrow,np.inf)])),"
        "np.ascontiguousarray(st[0],dtype=np.int8),"
        "np.ascontiguousarray(st[1],dtype=np.int64),"
        "1e-9,100000,%r);" % float(tl)
        + "print('RES '+json.dumps({'status':out[0],'obj':out[2],'iters':int(out[3])}))"
    )
    p = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True)
    assert p.returncode == 0, f"{path}: child failed rc={p.returncode}\n{p.stderr[-3000:]}"
    res = [json.loads(ln[4:]) for ln in p.stdout.splitlines() if ln.startswith("RES ")]
    assert len(res) == 1, f"{path}: expected 1 RES line, got {len(res)}\n{p.stdout[-2000:]}"
    phases = {}
    for ln in p.stderr.splitlines():
        m = PHASE.match(ln)
        if m:
            phases[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    return res[0], phases


paths = sorted(glob.glob(os.path.join(WT, "scratchpad/i1008/lps/*.npz")))
assert paths, "no captured LPs"
tl = float(os.environ.get("I1008_TL", "45"))

print(
    f"{'tag':22s} {'total_ms':>9s} {'symb_ms':>9s} {'num_ms':>9s} "
    f"{'nsymb':>5s} {'symb%':>6s} {'num%':>6s} {'other%':>6s}",
    flush=True,
)
parsed = 0
for p in paths:
    tag = os.path.basename(p)[:-4]
    res, ph = run_one(p, tl)
    assert "LuSymbolic" in ph, f"{tag}: no LuSymbolic phase in the dump — probe measured nothing"
    assert "LuNumeric" in ph, f"{tag}: no LuNumeric phase in the dump"
    nsym, symb = ph["LuSymbolic"]
    nnum, num = ph["LuNumeric"]
    total = ph["DualPivotLoop"][1] + ph.get("DualPrepare", (0, 0.0))[1]
    assert total > 0, f"{tag}: zero total time"
    print(
        f"{tag:22s} {total:9.1f} {symb:9.1f} {num:9.1f} {nsym:5d} "
        f"{100 * symb / total:5.1f}% {100 * num / total:5.1f}% "
        f"{100 * (total - symb - num) / total:5.1f}%",
        flush=True,
    )
    print(
        "JSON3 "
        + json.dumps(
            {
                "tag": tag,
                "total_ms": total,
                "symb_ms": symb,
                "num_ms": num,
                "n_symb": nsym,
                "n_num": nnum,
                "status": res["status"],
                "iters": res["iters"],
            }
        ),
        flush=True,
    )
    parsed += 1

print("parsed solves:", parsed, flush=True)
if parsed == 0:
    sys.exit(1)
