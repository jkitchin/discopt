"""#1116 stage 4: is the root relaxation BUILD being truncated by a wall deadline?

Usage: python -u build_truncation_probe.py <instance-stem> <max_nodes> [n_calls]

``build_component_bisect.py`` showed the first root LP has a DIFFERENT NUMBER OF
COLUMNS in two repetitions of the same solve in the same process — rep0
``(4860, 1532)`` vs rep1 ``(4734, 1469)``. That is not float noise and not
iteration order: a different relaxation was built.

The #694 "anytime build" is the mechanism that can do exactly that.
``build_uniform_relaxation`` stops its constraint-row loop once ``build_deadline``
(a ``perf_counter`` time) is spent and records the fact on the model:
``_build_truncated`` / ``_build_constraints_done`` / ``_build_constraints_total``.
A build truncated at a machine-speed-dependent point yields fewer rows AND fewer
lifted columns — the observed signature.

This probe reads those three provenance fields (plus shapes) at every
``MilpRelaxationModel.solve`` and compares two repetitions.

Kill criterion: if ``_build_truncated`` is False on every call in both reps while
the shapes still differ, deadline truncation is NOT the cause and this direction
is dead — the differing column count then has to come from upstream (presolve /
term detection) instead.

Per-rep progress (§10), module assertion (§8), executed-comparison count with a
non-zero exit when it is zero (§6), no swallowed exceptions (§7).
"""

import json
import sys

import discopt
import scipy.sparse as sp
from discopt._relax.milp_relaxation import MilpRelaxationModel
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem = sys.argv[1]
max_nodes = int(sys.argv[2])
N_CALLS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

runs: list[list[dict]] = []
_cur: list[dict] = []
_orig = MilpRelaxationModel.solve


def _traced(self, *a, **kw):
    if len(_cur) < N_CALLS:
        A = self._A_ub
        shape = None if A is None else tuple(sp.csr_matrix(A).shape)
        nnz = None if A is None else int(sp.csr_matrix(A).nnz)
        _cur.append(
            {
                "call": len(_cur),
                "n_cols": int(len(self._c)),
                "shape": shape,
                "nnz": nnz,
                "build_truncated": bool(getattr(self, "_build_truncated", False)),
                "cons_done": getattr(self, "_build_constraints_done", None),
                "cons_total": getattr(self, "_build_constraints_total", None),
            }
        )
        print(json.dumps({"rep": len(runs), **_cur[-1]}), flush=True)
    return _orig(self, *a, **kw)


MilpRelaxationModel.solve = _traced

for rep in range(2):
    _cur = []
    model = from_nl(NL.format(stem))
    r = model.solve(max_nodes=max_nodes)
    runs.append(_cur)
    print(
        f"rep={rep} calls={len(_cur)} nodes={r.node_count} bound={r.bound!r} "
        f"objective={r.objective!r}",
        flush=True,
    )

A, B = runs
comparisons = 0
truncated_seen = 0
for k in range(min(len(A), len(B))):
    comparisons += 1
    a, b = A[k], B[k]
    truncated_seen += int(a["build_truncated"]) + int(b["build_truncated"])
    same = a["shape"] == b["shape"] and a["n_cols"] == b["n_cols"] and a["nnz"] == b["nnz"]
    print(
        f"CALL {k}: {'SAME' if same else 'DIFFERS'}  "
        f"rep0 cols={a['n_cols']} shape={a['shape']} nnz={a['nnz']} "
        f"trunc={a['build_truncated']} done={a['cons_done']}/{a['cons_total']}  |  "
        f"rep1 cols={b['n_cols']} shape={b['shape']} nnz={b['nnz']} "
        f"trunc={b['build_truncated']} done={b['cons_done']}/{b['cons_total']}",
        flush=True,
    )

print(f"comparisons={comparisons} truncated_flags_seen={truncated_seen}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
