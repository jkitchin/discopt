"""#1116 first-divergence bisection at the LP seam.

Usage: python -u lp_seam_bisect.py <instance-stem> <max_nodes>

Solves the SAME instance twice in ONE process, recording for every
``MilpRelaxationModel.solve`` call, in call order, a hash of the LP *input*
(c, A_ub, b_ub, bounds, integrality) and of the LP *output* (status, objective,
x). Then it reports the FIRST call index at which the two runs disagree, and
which side disagreed:

* inputs already differ  -> the divergence happened UPSTREAM of the LP (build,
  separation, branching, bound propagation); the LP is a faithful function.
* inputs match, outputs differ -> the LP solver itself is not a function of its
  input (a genuine simplex-level nondeterminism).

Prints per-rep progress (CLAUDE.md §10), asserts which module it loaded (§8),
counts executed comparisons and exits non-zero when it made none (§6). No
exception is swallowed anywhere in the instrument (§7).
"""

import hashlib
import sys

import discopt
import numpy as np
import scipy.sparse as sp
from discopt._relax.milp_relaxation import MilpRelaxationModel
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes = sys.argv[1], int(sys.argv[2])

_MAXLEN = 400_000  # cap per-call hashing cost; layout+prefix is plenty to separate runs


def _h(*parts) -> str:
    d = hashlib.blake2b(digest_size=12)
    for p in parts:
        if p is None:
            d.update(b"\x00None")
            continue
        if sp.issparse(p):
            m = sp.csr_matrix(p)
            d.update(b"\x01sp")
            d.update(np.asarray(m.shape, dtype=np.int64).tobytes())
            for arr in (m.indptr, m.indices, m.data):
                a = np.ascontiguousarray(arr)
                d.update(a.tobytes()[:_MAXLEN])
            continue
        a = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
        d.update(b"\x02np")
        d.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        d.update(a.tobytes()[:_MAXLEN])
    return d.hexdigest()


trace: list[tuple[str, str]] = []
_orig = MilpRelaxationModel.solve


def _traced(self, *a, **kw):
    bounds = np.asarray(self._bounds, dtype=np.float64) if self._bounds else None
    hin = _h(self._c, self._A_ub, self._b_ub, bounds, self._integrality)
    res = _orig(self, *a, **kw)
    hout = _h(
        np.asarray([float(res.objective) if res.objective is not None else np.nan]),
        getattr(res, "x", None),
    )
    trace.append((hin, f"{res.status}|{hout}"))
    return res


MilpRelaxationModel.solve = _traced

runs = []
for rep in range(2):
    trace = []
    model = from_nl(NL.format(stem))
    r = model.solve(max_nodes=max_nodes)
    runs.append(trace)
    print(
        f"rep={rep} lp_calls={len(trace)} nodes={r.node_count} "
        f"bound={r.bound!r} objective={r.objective!r} status={r.status}",
        flush=True,
    )

a, b = runs
comparisons = 0
first_in = None
first_out = None
for i in range(min(len(a), len(b))):
    comparisons += 1
    if a[i][0] != b[i][0] and first_in is None:
        first_in = i
    if a[i][0] == b[i][0] and a[i][1] != b[i][1] and first_out is None:
        first_out = i
    if first_in is not None:
        break

print(f"lp_calls rep0={len(a)} rep1={len(b)}", flush=True)
print(f"first_differing_INPUT  = {first_in}", flush=True)
print(f"first_differing_OUTPUT_at_equal_input = {first_out}", flush=True)
if first_in is None and first_out is None:
    print("VERDICT: LP seam is IDENTICAL on every compared call", flush=True)
elif first_out is not None and (first_in is None or first_out < first_in):
    print("VERDICT: the LP SOLVER is not a function of its input", flush=True)
else:
    print(f"VERDICT: divergence is UPSTREAM of the LP, first at call {first_in}", flush=True)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
